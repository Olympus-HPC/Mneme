#include <cstdio>
#include <chrono>
#include <utility>  // std::pair
#include <cuda.h>
#include <curand_kernel.h>

// philox generates 128 bits of randomness at a time. 
// Kernel uses this explicitly by putting suitably transformed result into float4
// for all members of float4 to be consumed UNROLL has to be 4. Don't change!
const int UNROLL = 4;

template <typename scalar_t,
          typename accscalar_t,
          typename IndexType,
          typename mask_t>
__global__ void
fused_dropout_kernel(
  const scalar_t *__restrict__ a,
        scalar_t *__restrict__ b,
         mask_t *__restrict__ c,
  IndexType totalElements,
  accscalar_t p,
  std::pair<uint64_t, uint64_t> seeds) 
{
  accscalar_t scale = accscalar_t(1)/p;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) *
                           blockDim.x * gridDim.x * UNROLL;

  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x * UNROLL) {

    float4 rand = curand_uniform4(&state);
    scalar_t src[UNROLL];
    rand.x = rand.x < p;
    rand.y = rand.y < p;
    rand.z = rand.z < p;
    rand.w = rand.w < p;

    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        const IndexType aOffset = li;
        src[ii] = a[aOffset];
      }
    }

    #pragma unroll
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        const IndexType bOffset = li;
        b[bOffset] = src[ii]*(&rand.x)[ii]*scale;
        c[bOffset] = (mask_t)(&rand.x)[ii];
      }
    }
    __syncthreads();
  }
}

// aligned vector generates vectorized load/store
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

template <typename scalar_t,
          typename accscalar_t,
          typename IndexType,
          typename mask_t,
          int VEC>
__global__ void
fused_dropout_kernel_vec(
  const scalar_t *__restrict__ a,
        scalar_t *__restrict__ b,
         mask_t *__restrict__ c,
  IndexType totalElements,
  accscalar_t p,
  std::pair<uint64_t, uint64_t> seeds) 
{
  using LoadT = aligned_vector<scalar_t, VEC>;
  using MaskLoadT = aligned_vector<mask_t, VEC>;

  // Helps align the total number of times curand_uniform4 is called by each thread for the same totalElements
  bool gridxvec_loop_state = 0;

  accscalar_t scale = accscalar_t(1)/p;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  // Note: Vectorized loads means we'll stride each thread by an additional VEC factor, as we'll load VEC elements at a time
  for (IndexType linearIndex = idx * VEC;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * VEC) {

    scalar_t src[VEC];
    LoadT *value = reinterpret_cast<LoadT*>(&src);

    float4 rand;
    if ((VEC == 4) || (gridxvec_loop_state == 0)) {
      rand = curand_uniform4(&state);
    } else {
      // sets up the last two values we generated last iteration to be used this iteration.
      rand.x = rand.z;
      rand.y = rand.w;
      gridxvec_loop_state ^= 1;
    }
    rand.x = rand.x < p;
    rand.y = rand.y < p;
    if (VEC == 4) {
      rand.z = rand.z < p;
      rand.w = rand.w < p;
    }

    *value = *reinterpret_cast<const LoadT*>(&a[linearIndex]);

    scalar_t r[VEC];
    mask_t mask[VEC];

    // Perform the actual computation
    #pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      r[ii] = src[ii]*(&rand.x)[ii]*scale;
      mask[ii] = (mask_t)(&rand.x)[ii];
    }
    // Vectorized writes for both mask & result
    *(reinterpret_cast<LoadT*>(&b[linearIndex])) = *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&c[linearIndex])) = *reinterpret_cast<MaskLoadT*>(&mask[0]);

    __syncthreads();
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t nelem = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  const int64_t block_size = 256;

  dim3 dim_block(block_size);
  dim3 grid_vec1(512);
  dim3 grid_vec2(grid_vec1.x / 2);
  dim3 grid_vec4(grid_vec1.x / 4);

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  rng_engine_inputs.first =  12345678;
  rng_engine_inputs.second = 87654321;

  int64_t self_size = nelem * sizeof(float);
  int64_t ret_size = self_size;
  int64_t mask_size = nelem * sizeof(uint8_t);

  float *self_info = (float*) malloc (self_size); 
  float *ret_info = (float*) malloc (ret_size); 
  uint8_t *mask_info = (uint8_t*) malloc (mask_size);

  for (int64_t i = 0; i < nelem; i++) {
    self_info[i] = 0.1f;
  }

  float *d_self_info;
  cudaMalloc((void**)&d_self_info, self_size); 
  cudaMemcpy(d_self_info, self_info, self_size, cudaMemcpyHostToDevice);

  float *d_ret_info;
  cudaMalloc((void**)&d_ret_info, ret_size); 

  uint8_t *d_mask_info;
  cudaMalloc((void**)&d_mask_info, mask_size);

  auto start = std::chrono::steady_clock::now();
  for (int p = 1; p <= repeat; p++) {
    float pa = (float)p / repeat;
    fused_dropout_kernel<float, float, unsigned int, uint8_t>
      <<<grid_vec1, dim_block>>>(d_self_info, d_ret_info, d_mask_info, 
                                 nelem, pa, rng_engine_inputs);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (VEC1) %lf (s)\n", time * 1e-9f);
 
  start = std::chrono::steady_clock::now();
  for (int p = 1; p <= repeat; p++) {
    float pa = (float)p / repeat;
    fused_dropout_kernel_vec<float, float, unsigned int, uint8_t, 2>
      <<<grid_vec2, dim_block>>>(d_self_info, d_ret_info, d_mask_info, 
                                 nelem, pa, rng_engine_inputs);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (VEC2) %lf (s)\n", time * 1e-9f);

  start = std::chrono::steady_clock::now();
  for (int p = 1; p <= repeat; p++) {
    float pa = (float)p / repeat;
    fused_dropout_kernel_vec<float, float, unsigned int, uint8_t, 4>
      <<<grid_vec4, dim_block>>>(d_self_info, d_ret_info, d_mask_info, 
                                 nelem, pa, rng_engine_inputs);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (VEC4) %lf (s)\n", time * 1e-9f);

  cudaFree(d_self_info); 
  cudaFree(d_ret_info); 
  cudaFree(d_mask_info); 
  free(self_info); 
  free(ret_info); 
  free(mask_info); 

  return 0;
}
