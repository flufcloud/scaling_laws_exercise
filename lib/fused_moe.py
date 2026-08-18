import functools

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def get_raw_expert_and_block(kernel_idx, g, BLOCK_B,):
  # reimplementation of GroupInfo method, returns the  rows, expert_id, real start_idx, real_end_idx
  block_size = jnp.array(BLOCK_B, jnp.int32)
  start = end = group_start = group_end = block = group = jnp.array(0, dtype=jnp.int32)

  # we need to do this for jit
  boundaries = [g[i] for i in range(g.shape[0])]

  # we need to iterate over the expert counts to calculate what the given kernel_idx's
  # start_index, end_index, block, group, and size are
  # the key insight is that we have at most len(boundaries) + (rows // BLOCK_B)
  # threads. So we can go through and calculate the indices
  # One note is that for perfectly split expert counts (ie if for some expert count the start_index and end_indxed perfectly fit into a block)
  # we need to account for a 'dead' thread. This is because we assume a priori that we will launch len(boundaries) - 1 threads
  # so even if we an expert count perfectly into memory, we still need to launch an extra thread to make sure the number of threads we launch is constant
  for i, e_count in enumerate(boundaries):
    # calculates the start and end index
    start = end
    end = start + e_count

    start_block = start // block_size
    # we have to do this because we want to find the block following the block that the final element is in
    # this is important for calculating how many threads will be assigned per expert
    # this also works in the case that end = 0 (ie 0 e_count): it sets the start and end block to 0, so later all kernel_idxs will necessarily
    # avoid computation for this
    end_block = ((end - 1) // block_size) + 1

    # get the half open interval of [start_idx, end_idx) of threads that are responsible for computing this experts values
    # we add i to account for extra threads launched at the expert boundaries
    start_idx = start_block + i
    end_idx = end_block + i

    # this tells us if the kernel_idx is one of the idxs responsible for computing this experts counts
    is_group = (kernel_idx >= start_idx) & (kernel_idx < end_idx)

    # is_group is only true for at most one iteration of the loop, so we only set these once
    # block here means the actual block of memory that the thread should access
    # the kernel_idx is i steps ahead of the actual block it's in, so we offset it by i to get the block
    block = jax.lax.select(is_group, kernel_idx - i, block)
    group = jax.lax.select(is_group, jnp.array(i, jnp.int32), group)

    # we need to get the start indices and end indices of the group this thread is responsible for
    group_start = jax.lax.select(is_group, start, group_start)
    group_end = jax.lax.select(is_group, end, group_end)

  # We now have the relevant block, group_start, and group_end values for this kernel idx
  # now we have to calculate the actual indices within the block that this idx is responsible for
  block_start = block * block_size
  real_start = jnp.maximum(block_start, group_start)

  block_end = block_start + block_size
  real_end = jnp.minimum(block_end, group_end)
  rows = real_end - real_start
  relative_start = real_start - block_start
  relative_end = relative_start + rows
  return rows, group, block, relative_start, relative_end

def update_dead_threads(rows):
  valid = rows != 0
  idx = jnp.arange(rows.shape[0])
  last_valid_idx = jax.lax.cummax(jnp.where(valid, idx, -1))
  return last_valid_idx

vmapped_expert_metadata = jax.vmap(get_raw_expert_and_block, in_axes=(0, None, None))
def get_expert_metadata(kernel_idx, g, BLOCK_B):
  rows, group, block, relative_start, relative_end = vmapped_expert_metadata(kernel_idx, g, BLOCK_B)
  last_valid_idx = update_dead_threads(rows)
  return rows, group[last_valid_idx], block[last_valid_idx], relative_start, relative_end


def fused_matmul_kernel(metadata, x_ref, W_up_ref, W_down_ref, b_up_ref, b_down_ref, y_ref,
                        *, BLOCK_B, F_BLOCKS):

  # the batch mapping is given by i which is the the middle grid dim
  pid = pl.program_id(1)
  @pl.when(metadata[0][pid] != 0)
  def _():
      relative_start_index, relative_end_index = metadata[3][pid], metadata[4][pid]
      expert = metadata[1][pid]
      mask = jax.lax.broadcasted_iota(
        dtype=jnp.int32,
        shape=(BLOCK_B, 1),
        dimension=0,)
      mask = (mask >= relative_start_index) & (mask < relative_end_index)

      @pl.when(pl.program_id(2) == 0)
      def _():
        y = y_ref[...]
        y_ref[...] = jnp.where(mask, 0, y)

      @pl.when(pl.program_id(2) != F_BLOCKS - 1)
      def _():
        x = x_ref[...] @ W_up_ref[0][...]
        x = x + b_up_ref[expert][...]
        x = jnp.where(mask, x, 0)

        x = jax.nn.gelu(x)
        x = x @ W_down_ref[0][...]
        y_ref[...] += x

      @pl.when(pl.program_id(2) == F_BLOCKS - 1)
      def _():
        # repeating lines of code to avoid loading y_ref more than once. TODO: profile to check whether this actually helps
        x = x_ref[...] @ W_up_ref[0][...]
        x = x + b_up_ref[expert][...]
        x = jnp.where(mask, x, 0)

        x = jax.nn.gelu(x)
        x = x @ W_down_ref[0][...]

        x = x + b_down_ref[expert][...]
        x = jnp.where(mask, x, 0)
        y_ref[...] += x

def fused_matmul(X, W_up, W_down, b_up, b_down, g,*, BLOCK_B, BLOCK_D, BLOCK_F):
  # E must be divisible by 8 for TPU block constraints when indexing into bias
  E = W_up.shape[0]
  D = W_up.shape[1]
  F = W_down.shape[1]
  B = X.shape[0]
  B_BLOCKS = (B // BLOCK_B) + g.shape[0] - 1
  D_BLOCKS = D // BLOCK_D
  F_BLOCKS = F // BLOCK_F
  blocks = (D_BLOCKS,
            B_BLOCKS,
            F_BLOCKS,)

  metadata = get_expert_metadata(jax.lax.iota(jnp.int32, B_BLOCKS), g, BLOCK_B)

  grid_spec = pltpu.PrefetchScalarGridSpec(grid=blocks,
                                            num_scalar_prefetch=1,
                                            in_specs=[
                                                pl.BlockSpec((BLOCK_B, D), lambda j, i, k, met: (met[2][i], 0)),
                                                pl.BlockSpec((1, D, BLOCK_F), lambda j, i, k, met: (met[1][i], 0, k)),
                                                pl.BlockSpec((1, BLOCK_F, BLOCK_D), lambda j, i, k, met: (met[1][i], k, j)),
                                                pl.BlockSpec((E, BLOCK_F), lambda j, i, k, met: (0, k)),
                                                pl.BlockSpec((E, BLOCK_D), lambda j, i, k, met: (0, j)),
                                            ],
                                            out_specs=pl.BlockSpec((BLOCK_B, BLOCK_D), lambda j, i, k, met: (met[2][i], j)),
                                            )
  return pl.pallas_call(
      functools.partial(fused_matmul_kernel, BLOCK_B=BLOCK_B, F_BLOCKS=F_BLOCKS,),
      jax.ShapeDtypeStruct(X.shape, X.dtype),
      grid_spec=grid_spec,
      interpret=False,
  )(metadata, X, W_up, W_down, b_up, b_down)
