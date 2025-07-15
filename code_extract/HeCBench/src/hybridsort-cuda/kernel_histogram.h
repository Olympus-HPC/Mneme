////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute per-workgroup partial histograms
////////////////////////////////////////////////////////////////////////////////
  __global__ void
histogram1024 ( unsigned int* histoOutput, 
    const float* histoInput,
    const int listsize,
    const float minimum,
    const float maximum)
{
  __shared__ unsigned int s_Hist[HISTOGRAM_BLOCK_MEMORY]; 

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lid = threadIdx.x;
  const int gsize = gridDim.x * blockDim.x;
  const int lsize = blockDim.x;

  //Per-warp substorage storage
  int mulBase = (lid >> BUCKET_WARP_LOG_SIZE);
  const int warpBase = IMUL(mulBase, HISTOGRAM_BIN_COUNT);

  //Clear shared memory storage for current threadblock before processing
  for(uint i = lid; i < HISTOGRAM_BLOCK_MEMORY; i+=lsize) {
    s_Hist[i] = 0;
  }


  //Read through the entire input buffer, build per-warp histograms
  __syncthreads();
  for(int pos = gid; pos < listsize; pos += gsize) {
    uint data4 = ((histoInput[pos] - minimum)/(maximum - minimum)) * HISTOGRAM_BIN_COUNT;

    atomicAdd(&s_Hist[warpBase+(data4 & 0x3FFU)], 1U);
  }

  //Per-block histogram reduction
  __syncthreads();

  for(int pos = lid; pos < HISTOGRAM_BIN_COUNT; pos += lsize){
    uint sum = 0;
    for(int i = 0; i < HISTOGRAM_BLOCK_MEMORY; i+= HISTOGRAM_BIN_COUNT){ 
      sum += s_Hist[pos + i] & 0x07FFFFFFU;
    }
    atomicAdd(&histoOutput[pos], sum);
  }
}

