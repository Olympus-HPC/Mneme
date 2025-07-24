#include <climits>
#include <cstdio>

#include "mneme/DeviceTraits.hpp"
using namespace mneme;

#ifdef MNEME_ENABLE_HIP
using MnemeDeviceRT = DeviceTraits<DeviceVendors::HIP>;
#elif defined(MNEME_ENABLE_CUDA)
using MnemeDeviceRT = DeviceTraits<DeviceVendors::CUDA>;
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

//=== Replace these with your allocator’s API ===//
void *gpu_alloc(std::size_t bytes) {
  void *ptr;
  std::cout << "Requesting " << bytes << " " << bytes / (1024.0 * 1024.0)
            << "MB" << "\n";
  auto EC =
      MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceMalloc(&ptr, bytes));
  if (EC) {
    std::cout << "Fatal error with malloc value " << EC.value() << "\n";
  }
  return ptr;
}
void gpu_free(void *ptr) {
  std::cout << "Releasing " << ptr << "\n";
  auto EC = MnemeDeviceRT::DeviceErrorCheck(MnemeDeviceRT::DeviceFree(ptr));
  if (EC) {
    std::cout << "Fatal error with free value " << EC.value() << "\n";
  }
}
//================================================//

int main(int argc, char *argv[]) {
  // Parse arguments (with defaults)
  int iterations = 50;
  int K = 20;
  std::size_t max_size = 1 << 30; // 1 GB
  // before the loop
  double min_log = std::log(1.0);                           // ln(1 B)
  double max_log = std::log(static_cast<double>(max_size)); // ln(1 GB)
  std::uniform_real_distribution<double> log_dist(min_log, max_log);
  int seed = 0;

  if (argc > 1)
    iterations = std::stoi(argv[1]);
  if (argc > 2)
    K = std::stoi(argv[2]);
  if (argc > 3)
    max_size = std::stoull(argv[3]);
  if (argc > 4)
    seed = std::stoi(argv[4]);

  // rng seeded from high‑resolution clock
  std::mt19937_64 rng(seed);

  std::vector<void *> live;

  for (int iter = 0; iter < iterations; ++iter) {
    std::cout << "Running iteration " << iter << "\n";
    // 1) K allocations
    for (int i = 0; i < K; ++i) {
      // inside your allocation loop
      double x = log_dist(rng); // uniform in [ln(1), ln(1GB)]
      size_t sz = static_cast<size_t>(std::exp(x));
      // clamp in case of rounding
      sz = std::max<size_t>(1, std::min(sz, max_size));
      void *p = gpu_alloc(sz);
      if (!p) {
        std::cerr << "Allocation failed at iteration " << iter << ", index "
                  << i << " (size=" << sz << ")\n";
        // decide: exit, throw, or continue?
        return 1;
      }
      live.push_back(p);
    }

    // 2) D random deallocations (if enough live pointers)
    std::uniform_int_distribution<std::size_t> num_free(0, live.size());
    int can_free = num_free(rng);
    std::cout << "Freeing " << can_free << " out of " << live.size() << "\n";
    if (can_free > 0) {
      std::uniform_int_distribution<std::size_t> idx_dist(0, live.size() - 1);
      // pick `can_free` distinct indices via Fisher–Yates “partial shu
      // fle”
      for (int i = 0; i < can_free; ++i) {
        std::size_t j = idx_dist(rng);
        std::swap(live[i], live[j]);
      }
      // free the first `can_free`
      for (int i = 0; i < can_free; ++i) {
        gpu_free(live[i]);
      }
      // remove them from the vector
      live.erase(live.begin(), live.begin() + can_free);
    }
    std::cout << "After release i have " << live.size() << "\n";
  }

  // 3) tear down: free any remaining
  for (void *p : live) {
    gpu_free(p);
  }
  live.clear();

  return 0;
}
