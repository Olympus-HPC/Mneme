#include <iostream>
#include <cstdio>
#include <chrono>
#include <cuda.h>

#define NOW std::chrono::high_resolution_clock::now()

#include "fastdiv.h"
#include "kernels.h"

// Functional test returns 1 when it fails; otherwise it returns 0
int test()
{
  const int blocks = 256;
  const int divisor_count = 100000;
  const int divident_count = 1000000;

  int grids = (divident_count + blocks - 1) / blocks;

  int buf[4];
  int * buf_d;
  cudaMalloc(&buf_d, sizeof(int) * 4);

  std::cout << "Running functional test on " << divisor_count << " divisors, with " 
            << grids * blocks << " dividents for each divisor" << std::endl;

  for(int d = 1; d < divisor_count; ++d)
  {
    for(int sign = 1; sign >= -1; sign -= 2)
    {
      int divisor = d * sign;
      cudaMemset(buf_d, 0, sizeof(int) * 4);
      check<<<grids, blocks>>>(divisor, buf_d);
      cudaMemcpy(buf, buf_d, sizeof(int) * 4, cudaMemcpyDeviceToHost);

      if (buf[0] > 0)
      {
        std::cout << buf[0] << " wrong results, one of them is for divident " 
                  << buf[1] << ", correct quotient = " << buf[2] 
                  << ", fast computed quotient = " << buf[3] << std::endl;
        cudaFree(buf_d);
        return 1;
      }
    }
  }

  cudaFree(buf_d);
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // performance evaluation after functional test is done
  if (test()) return 1;

  const int grids = 32 * 1024;
  const int blocks = 256;

  // warmup may be needed for accurate performance measurement with chrono
  for (int i = 0; i < 100; i++) {
    throughput_test<int><<<grids, blocks>>>(3, 5, 7, 0, 0);
    throughput_test<int_fastdiv><<<grids, blocks>>>(3, 5, 7, 0, 0);
  }
  cudaDeviceSynchronize();

  std::cout << "THROUGHPUT TEST" << std::endl;

  std::cout << "Benchmarking plain division by constant... ";
  auto start = NOW;

  for (int i = 0; i < repeat; i++)
    throughput_test<int><<<grids, blocks>>>(3, 5, 7, 0, 0);
  cudaDeviceSynchronize();

  auto end = NOW;
  std::chrono::duration<double> elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++)
    throughput_test<int_fastdiv><<<grids, blocks>>>(3, 5, 7, 0, 0);
  cudaDeviceSynchronize();

  end = NOW;
  std::chrono::duration<double> elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;

  // warmup
  for (int i = 0; i < 100; i++) {
    latency_test<int><<<grids, blocks>>>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
    latency_test<int_fastdiv><<<grids, blocks>>>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
  }
  cudaDeviceSynchronize();

  std::cout << "LATENCY TEST" << std::endl;
  std::cout << "Benchmarking plain division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++)
    latency_test<int><<<grids, blocks>>>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
  cudaDeviceSynchronize();

  end = NOW;
  elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++)
    latency_test<int_fastdiv><<<grids, blocks>>>(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
  cudaDeviceSynchronize();

  end = NOW;
  elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;
  return 0;
}
