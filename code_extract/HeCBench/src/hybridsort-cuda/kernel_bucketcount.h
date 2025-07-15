
  __global__ void
bucketcount (const float* input , 
    int* indice,
    unsigned int* prefixoffsets,
    const float* pivotpoints,
    const int listsize)
{
  __shared__ unsigned int s_offset[BUCKET_BLOCK_MEMORY]; 

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int gsize = gridDim.x * blockDim.x;
  const int lsize = blockDim.x;
  const int warpBase = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
  const int numThreads = gsize;

  for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
    s_offset[i] = 0;

  __syncthreads();

  for (int tid = gid; tid < listsize; tid += numThreads) {
    float elem = input[tid];

    int idx  = DIVISIONS/2 - 1;
    int jump = DIVISIONS/4;
    float piv = pivotpoints[idx]; //s_pivotpoints[idx];

    while(jump >= 1){
      idx = (elem < piv) ? (idx - jump) : (idx + jump);
      piv = pivotpoints[idx]; //s_pivotpoints[idx];
      jump /= 2;
    }
    idx = (elem < piv) ? idx : (idx + 1);

    indice[tid] = 
      (atomicAdd(&s_offset[warpBase+idx], 1U) << LOG_DIVISIONS)  + idx;
  }

  __syncthreads();

  int prefixBase = blockIdx.x * BUCKET_BLOCK_MEMORY;

  for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
    prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;

}

