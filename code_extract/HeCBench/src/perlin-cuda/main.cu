// Based on the Perlin noise generator written by G. Parolini && I. Cislaghi

#include <iostream>
#include <unistd.h>
#include "utils.hpp"
#include "noise.hpp"

int main(int argc, char **argv) {

  NoiseParams params;
  // set default values for the parameters
  params.ppu = 250.f;
  params.seed = 0;
  params.octaves = 3;
  params.lacunarity = 2;
  params.persistence = 0.5;

  uint8_t *hPixels;
  CHECK(cudaMallocHost((void**)&hPixels, 4 * WIN_WIDTH * WIN_HEIGHT * sizeof(uint8_t)));

  for (int nStreams = 1; nStreams <= 32; nStreams *= 2) {

    std::cout << std::endl;
    std::cout << "Using " << nStreams << " streams." << std::endl;

    Perlin perlin;

    cudaStream_t *streams = new cudaStream_t[nStreams];
    for (int i = 0; i < nStreams; ++i) {
      CHECK(cudaStreamCreate(&streams[i]));
    }

    perlin.calculate(hPixels, params, streams, nStreams);

    uint64_t checksum = 0;
    for (uint64_t i = 0; i < 4 * WIN_WIDTH * WIN_HEIGHT; ++i)
      checksum += hPixels[i];
    std::cout << "checksum = " << checksum / (4 * WIN_WIDTH * WIN_HEIGHT) << std::endl;

    for (int i = 0; i < nStreams; ++i) {
      CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
  }

  CHECK(cudaFreeHost(hPixels));

  return 0;
}
