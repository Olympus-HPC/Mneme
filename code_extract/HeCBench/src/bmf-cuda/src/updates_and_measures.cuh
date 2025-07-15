#ifndef UPDATES_CUH
#define UPDATES_CUH

#include "helper/cuda_helpers.cuh"
#include "helper/rngpu.hpp"

// uint32_t vector masks --------------------------------------------------------
__inline__ HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask_many(const uint8_t factorDim, fast_kiss_state32_t state, const uint32_t rand_depth) {
  uint32_t bit_flip_mask = FULLMASK >> (32-factorDim);
  #pragma unroll
  for(unsigned int i = 0; i < rand_depth; ++i) {
    bit_flip_mask &= fast_kiss32(state);
  }
  // bit_flip_mask &= FULLMASK >> (32-factorDim);
  return bit_flip_mask;
}

__inline__ HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask_all(const uint8_t factorDim) {
  return FULLMASK >> (32-factorDim);
}

__inline__ HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask_one(const uint8_t factorDim, fast_kiss_state32_t state) {
  const uint32_t lane = fast_kiss32(state) % factorDim;
  return 1 << lane;
}

__inline__ HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask(const uint8_t factorDim, fast_kiss_state32_t state,
    const float flipManyChance,
    const uint32_t flipManyDepth) {
  const float random_many = fast_kiss32(state) / (float) UINT32_MAX;

  return random_many < flipManyChance ? get_flip_mask_many(factorDim, state, flipManyDepth)
    : get_flip_mask_one(factorDim, state);
  // return random_many < flipManyChance ? get_flip_mask_all() : get_flip_mask_one(state);
}

// Metropolis–Hastings algorithm
template<typename error_t>
__inline__ HOST_DEVICE_QUALIFIER
bool metro(fast_kiss_state32_t state, const error_t error, const float temperature, const int error_max = 1) {
  if(error <= 0) return true;
  if(temperature <= 0) return false;
  const float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;
  // const float metro = fminf(1.0f, expf((float) - error / error_max / temperature));
  const float metro = expf((float) - error / error_max / temperature);
  return randomNumber < metro;
}


// error measures ---------------------------------------------------------------
// weight*false negatives + false positives
template<typename error_t>
__inline__ HOST_DEVICE_QUALIFIER
error_t error_measure(const int test, const int truth, const error_t weigth) {
  return (truth == 1) ? weigth * (test ^ truth) : (test ^ truth);
}

#endif
