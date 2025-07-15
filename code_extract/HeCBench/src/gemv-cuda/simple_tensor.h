#ifndef SIMPLE_TENSOR_H_
#define SIMPLE_TENSOR_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

#include <random>

template <typename T>
class SimpleTensor {
 public:
  SimpleTensor(unsigned height, unsigned width)
      : height_(height), width_(width) {
    cudaMalloc((void**)&data_, height_ * width_ * sizeof(T));
  }
  T* device_data() const { return data_; }
  /**
   * @brief generate a height_ * width_ matrix with random fp16 numbers
   */
  void reset();
  /**
   * @brief copy the numbers from device to the host
   */
  void to_host(T* host_data, unsigned n);
  /**
   * @brief move constructor
   */
  SimpleTensor(SimpleTensor&& other) noexcept
      : height_(other.height_), width_(other.width_), data_(other.data_) {
    other.data_ = nullptr;  // Ensure the other object won't delete the data
                            // after being destroyed
  }
  /**
   * @brief overload the assignment operator for move semantics
   */
  SimpleTensor& operator=(SimpleTensor&& other) noexcept {
    if (this != &other) {
      height_ = other.height_;
      width_ = other.width_;

      // Deallocate existing data
      cudaFree(data_);

      // Take ownership of the new data
      data_ = other.data_;
      other.data_ = nullptr;
    }

    return *this;
  }
  ~SimpleTensor() { cudaFree(data_); }

  unsigned int width_;
  unsigned int height_;
  // device data
  T* data_;
};

template <typename T>
void SimpleTensor<T>::reset() {
  unsigned int total_elements = height_ * width_;
  std::vector<T> rng (total_elements);
  std::mt19937 gen(19937);
  if constexpr (std::is_same<T, __half>::value) {
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (unsigned n = 0; n < total_elements; ++n)
      rng[n] = __float2half(dis(gen));
    cudaMemcpy(data_, rng.data(), total_elements * sizeof(T), cudaMemcpyHostToDevice);
  } else if constexpr (std::is_same<T, int8_t>::value) {
    std::uniform_int_distribution<int> dis(-128, 127);
    for (unsigned n = 0; n < total_elements; ++n)
      rng[n] = dis(gen);
    cudaMemcpy(data_, rng.data(), total_elements * sizeof(T), cudaMemcpyHostToDevice);
  } else if constexpr (std::is_same<T, uint4_2>::value) {
    std::uniform_int_distribution<int> dis(0, 15);
    for (unsigned n = 0; n < total_elements; ++n) {
      rng[n].setX(dis(gen));
      rng[n].setY(dis(gen));
    }
    cudaMemcpy(data_, rng.data(), total_elements * sizeof(T), cudaMemcpyHostToDevice);
  }
}

template <typename T>
void SimpleTensor<T>::to_host(T* host_data, unsigned n) {
  unsigned int total_elements = height_ * width_;
  assert(n <= total_elements);
  cudaMemcpy(host_data, data_, n * sizeof(T), cudaMemcpyDeviceToHost);
}

#endif  // SIMPLE_TENSOR_H_
