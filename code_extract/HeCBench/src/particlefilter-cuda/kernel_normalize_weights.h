__global__ void
kernel_normalize_weights (
    float* __restrict__ weights,
    const float* __restrict__ partial_sums,
    float* __restrict__ CDF,
    float* __restrict__ u,
    int* __restrict__ seed,
    const int Nparticles )
{
  __shared__ float u1;
  __shared__ float sumWeights;
  int local_id = threadIdx.x;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(0 == local_id)
    sumWeights = partial_sums[0];
  __syncthreads();
  if(i < Nparticles) {
    weights[i] = weights[i]/sumWeights;
  }
  __syncthreads();
  if(i == 0) {
    CDF[0] = weights[0];
    for(int x = 1; x < Nparticles; x++){
      CDF[x] = weights[x] + CDF[x-1];
    }

    seed[i] = (A*seed[i] + C) % M;
    float p = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    float q = fabsf(seed[i]/((float)M));
    u[0] = (1.0f/((float)(Nparticles))) * 
      (sqrtf(-2.0f*log(p))*cosf(2.0f*PI*q));
    // do this to allow all threads in all blocks to use the same u1
  }
  __syncthreads();
  if(0 == local_id)
    u1 = u[0];

  __syncthreads();
  if(i < Nparticles)
  {
    u[i] = u1 + i/((float)(Nparticles));
  }
}
