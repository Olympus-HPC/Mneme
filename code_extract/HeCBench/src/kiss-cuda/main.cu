#include <chrono>
#include <cstdio>
#include "kiss.cuh"

template <class T, class Rng>
__global__
void kernel (T * const out, const std::uint64_t n, const std::uint32_t seed)
{
  const std::uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  // generate initial local seed per thread
  const std::uint32_t local_seed =
      kiss::hashers::MurmurHash<std::uint32_t>::hash(seed+tid);

  Rng rng {local_seed};

  // grid-stride loop
  const auto grid_stride = blockDim.x * gridDim.x;
  for(std::uint64_t i = tid; i < n; i += grid_stride)
  {
      // generate random element and write to output
      out[i] = rng.template next<T>();
  }
}

template<class T, class Rng>
inline
void uniform_distribution(
    T * const out,
    const std::uint64_t n,
    const std::uint32_t seed) noexcept
{
  // execute kernel
  kernel<T, Rng><<<4096, 256>>>(out, n, seed);
}

// generation of gigabytes of uniform random values

int main(int argc, char* argv[])
{
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      return 1;
    }
    const int repeat = atoi(argv[1]);

    // define the data types to be generated
    using data_t = std::uint64_t;
    using rng_t = kiss::Kiss<data_t>;

    static constexpr std::uint64_t n = 1UL << 28;

    static constexpr std::uint32_t seed = 42;

    data_t * data_h = nullptr;
    cudaMallocHost(&data_h, sizeof(data_t)*n);

    data_t * data_d = nullptr;
    cudaMalloc(&data_d, sizeof(data_t)*n);

    cudaMemset(data_d, 0, sizeof(data_t)*n);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      uniform_distribution<data_t, rng_t>(data_d, n, seed);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

    cudaMemcpy(data_h, data_d, sizeof(data_t)*n, cudaMemcpyDeviceToHost);

    printf("\nThe first 10 random numbers:\n");
    for(std::uint64_t i = 0; i < 10; i++)
    {
      printf("%lu\n", data_h[i]);
    }

    cudaFreeHost(data_h);
    cudaFree(data_d);
    return 0;
}
