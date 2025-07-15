__global__ void
bucketprefix (
    unsigned int* prefixoffsets,
    unsigned int* offsets,
    int blocks )
{

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int size = blocks * BUCKET_BLOCK_MEMORY;
  int sum = 0;

  for (int i = tid; i < size; i += DIVISIONS) {
    int x = prefixoffsets[i];
    prefixoffsets[i] = sum;
    sum += x;
  }
  offsets[tid] = sum;
}

