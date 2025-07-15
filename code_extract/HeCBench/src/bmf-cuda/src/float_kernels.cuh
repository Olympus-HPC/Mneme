#ifndef FLOAT_KERNELS_CUH
#define FLOAT_KERNELS_CUH

// init kernel ---------------------------------------------------------------
__global__
void initFactor(
  float * A,
  const int height,
  const uint8_t factorDim,
  const uint32_t seed, 
  const float threshold)
{
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  const int warpLane = threadIdx.x % warpSize;

  if(warpId < height) {
    fast_kiss_state32_t state = get_initial_fast_kiss_state32(seed + tid);
    const int i = warpId;
    const int j = warpLane;

    A[i * factorDim + j] = j < factorDim ? fast_kiss32(state) < threshold : 0;
  }
}


// distance kernels ---------------------------------------------------------------
__global__
void computeDistanceRowsShared(
  const float * __restrict__ A,
  const float * __restrict__ B,
  const uint32_t * __restrict__ Cb, 
  const int height, const int width,
  const int padded_width,
  const uint8_t factorDim,
  const int inverse_density,
        int *__restrict__ global_error)
{
  const int warpIdIntern = threadIdx.x / warpSize;
  const int warpId = blockIdx.x * WARPSPERBLOCK + warpIdIntern;
  const int warpLane = threadIdx.x % warpSize;

  __shared__ int reductionArray[WARPSPERBLOCK];
  __shared__ float B_block[CHUNK_SIZE][32];
  __shared__ uint32_t C_block[CHUNK_SIZE];

  const uint32_t dim_mask = FULLMASK >> (32 - factorDim);

  const int i = warpId;
  const int k = warpLane;
  const bool A_i_k = A[i*warpSize + k] > 0.5f;

  const int vecRow = i / 32;
  const int vecFirst = vecRow * padded_width;
  const int vecLane = i % 32;
  int error_warp = 0;
  for (int j_chunk = 0; j_chunk < padded_width; j_chunk += CHUNK_SIZE) {
    #pragma unroll
    for(int j_local = warpIdIntern; j_local < CHUNK_SIZE; j_local += WARPSPERBLOCK) {
      const int j = j_chunk + j_local;
      B_block[j_local][k] = j < width ? B[j * warpSize + k] : 0;
    }
    if(threadIdx.x < CHUNK_SIZE) {
      const int vecId = vecFirst + j_chunk;
      C_block[threadIdx.x] = Cb[vecId + threadIdx.x];
    }
    __syncthreads();

    if (i < height) {
      #pragma unroll
      for(int j_local = 0; j_local < CHUNK_SIZE; ++j_local) {
        // int product = __any_sync(dim_mask, A_i_k && (B[j*warpSize + k] > 0.5f)) ? 1 : 0;
        const int product = __any_sync(dim_mask, A_i_k && (B_block[j_local][k] > 0.5f)) ? 1 : 0;

        const int C_ij = (C_block[j_local] >> vecLane) & 1;

        error_warp += error_measure(product, C_ij, inverse_density);
      }
    }
    __syncthreads();
  }

  if(warpLane == 0)
    reductionArray[warpIdIntern] = error_warp;
  __syncthreads();

  int error_block;
  if(warpIdIntern == 0) {
    error_block = warpReduceSum(reductionArray[warpLane], WARPSPERBLOCK);

    if (threadIdx.x == 0) {
      // Thread with threadIdx.x==0 now has total error of block
      atomicAdd(global_error, error_block);
    }
  }
}


#endif
