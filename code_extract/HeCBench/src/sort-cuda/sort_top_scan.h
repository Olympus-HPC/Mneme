__global__ void
top_scan (T* isums, const size_t num_work_groups)
{
  __shared__ T lmem[256*2];
  __shared__ T s_seed;
  int lid = threadIdx.x;
  int local_range = blockDim.x;

  if (lid == 0) s_seed = 0; 
  __syncthreads();

  // Decide if this is the last thread that needs to
  // propagate the seed value
  int last_thread = (lid < num_work_groups &&
                    (lid+1) == num_work_groups) ? 1 : 0;

  for (int d = 0; d < 16; d++)
  {
    T val = 0;
    // Load each block's count for digit d
    if (lid < num_work_groups)
    {
      val = isums[(num_work_groups * d) + lid];
    }
    // Exclusive scan the counts in local memory
    // T res = scanLocalMem(val, lmem, 1);
    int idx = lid;
    lmem[idx] = 0;
    idx += local_range;
    lmem[idx] = val;
    __syncthreads();
    for (int i = 1; i < local_range; i *= 2)
    {
      T t = lmem[idx -  i]; 
      __syncthreads();
      lmem[idx] += t;     
      __syncthreads();
    }
    T res = lmem[idx-1];

    // Write scanned value out to global
    if (lid < num_work_groups)
    {
      isums[(num_work_groups * d) + lid] = res + s_seed;
    }
    __syncthreads();

    if (last_thread)
    {
      s_seed += res + val;
    }
    __syncthreads();
  }
}
