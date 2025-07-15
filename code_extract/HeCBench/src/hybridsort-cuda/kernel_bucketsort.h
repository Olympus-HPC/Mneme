__global__ void
bucketsort (const float* input , 
    const int* indice,
    float* output,
    const unsigned int* prefixoffsets,
    const unsigned int* offsets,
    const int listsize)
{
  const int grp_id = blockIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int gsize = gridDim.x * blockDim.x;
  const int lsize = blockDim.x;

  __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY]; 

  int prefixBase = grp_id * BUCKET_BLOCK_MEMORY;
  const int warpBase = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
  const int numThreads = gsize;

  for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize){
    s_offset[i] = offsets[i & (DIVISIONS - 1)] + prefixoffsets[prefixBase + i];
  }

  __syncthreads();

  for (int tid = gid; tid < listsize; tid += numThreads){
    float elem = input[tid];
    int id = indice[tid];
    output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >>  LOG_DIVISIONS)] = elem;
  }
}

