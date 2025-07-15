__global__ void
reduce (const T* in, T* isums, const size_t size, const unsigned int shift)
{
  __shared__ T lmem[256];
  int group_range = gridDim.x;
  int group = blockIdx.x;
  int lid = threadIdx.x;
  int local_range = blockDim.x;

  int region_size = (size / 4) / group_range * 4;
  int block_start = group * region_size;

  // Give the last block any extra elements
  int block_stop  = (group == group_range - 1) ?
    size : block_start + region_size;

  // Calculate starting index for this thread/work item
  int tid = lid;
  int i = block_start + tid;

  // The per thread histogram, initially 0's.
  int digit_counts[16] = { 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0 };

  // Reduce multiple elements per thread
  while (i < block_stop)
  {
    // This statement
    // 1) Loads the value in from global memory
    // 2) Shifts to the right to have the 4 bits of interest
    //    in the least significant places
    // 3) Masks any more significant bits away. This leaves us
    // with the relevant digit (which is also the index into the
    // histogram). Next increment the histogram to count this occurrence.
    digit_counts[(in[i] >> shift) & 0xFU]++;
    i += local_range;
  }

  for (int d = 0; d < 16; d++)
  {
    // Load this thread's sum into local/shared memory
    lmem[tid] = digit_counts[d];
    __syncthreads();

    // Reduce the contents of shared/local memory
    for (unsigned int s = local_range / 2; s > 0; s >>= 1)
    {
      if (tid < s)
      {
        lmem[tid] += lmem[tid + s];
      }
      __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
      isums[d * group_range + group] = lmem[0];
    }
  }
}
