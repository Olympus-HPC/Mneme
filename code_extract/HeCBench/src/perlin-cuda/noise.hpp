#pragma once

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <cuda.h>
#include "noise_params.hpp"

class Perlin final {
  public:
    /** Calculates Perlin noise with given parameters and stores the result in `hPixels`.
     *  The computation is performed via the given CUDA streams.
     *  @return The timing statistics of the computation
     */
    void calculate(uint8_t *hPixels, NoiseParams params, cudaStream_t *streams, int nStreams);
};
