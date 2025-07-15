#include <cmath>
#include <chrono>
#include <iostream>
#include "noise.hpp"
#include "utils.hpp"

inline __device__ float lerp(float a, float b, float t)
{
  return a + t*(b-a);
}

inline __host__ __device__ float dot(float2 a, float2 b)
{
  return a.x * b.x + a.y * b.y;
}

// Precomputed 1 / sqrt(2)
static constexpr auto INVSQRT2 = .70710678118654752440;
static constexpr auto N_GRADIENTS = 8;

// Reference implementation: http://catlikecoding.com/unity/tutorials/noise/

// Pseudo-random permutation of int's as defined by Ken Perlin in his original reference implementation
__device__ const int _hash[256] = {
  151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
  140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
  247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
  57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
  74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
  60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
  65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
  200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
  52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
  207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
  119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
  129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
  218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
  81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
  184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
  222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180
};

// X and Y values of the "base gradients" are stored separately for optimization, as it's faster
// to pass around SoA than AoS
__device__ const float gradientX[N_GRADIENTS] = {
  1, -1, 0, 0, INVSQRT2, -INVSQRT2, INVSQRT2, -INVSQRT2
};

__device__ const float gradientY[N_GRADIENTS] = {
  0, 0, 1, -1, INVSQRT2, INVSQRT2, -INVSQRT2, -INVSQRT2
};

// This function (6t^5 - 15t^4 + 10t^3) has null first and second derivative at the "joint points"
// when used to calculate the distance between x point in the cell and its surrounding corners.
__inline__ __device__ float smooth(float t) {
  return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

/*
 * @param x Grid coordinate x
 * @param y Grid coordinate y (top-left = (0, 0))
 * @return Perlin noise at coordinate (x, y).
 */
__device__ float noiseAt(float x, float y, int seed) {

  // Get top-left corner indices
  const int ix = static_cast<int>(x),
            iy = static_cast<int>(y);

  // Weights
  const float wx = x - ix,
              wy = y - iy;

  // Get gradients at cell corners
  const int ix0 = ix & 255, iy0 = iy & 255;
  const int ix1 = (ix0 + 1) & 255, iy1 = (iy0 + 1) & 255;
  const int h0 = _hash[ix0], h1 = _hash[ix1];
  const int iTL = (_hash[h0 + iy0] + seed) % N_GRADIENTS,
            iTR = (_hash[h1 + iy0] + seed) % N_GRADIENTS,
            iBL = (_hash[h0 + iy1] + seed) % N_GRADIENTS,
            iBR = (_hash[h1 + iy1] + seed) % N_GRADIENTS;
  const float2 gTopLeft  = make_float2(gradientX[iTL], gradientY[iTL]);
  const float2 gTopRight = make_float2(gradientX[iTR], gradientY[iTR]);
  const float2 gBotLeft  = make_float2(gradientX[iBL], gradientY[iBL]);
  const float2 gBotRight = make_float2(gradientX[iBR], gradientY[iBR]);

  // Calculate dots between distance and gradient vectors
  const float dTopLeft  = dot(gTopLeft,  make_float2(wx,     wy));
  const float dTopRight = dot(gTopRight, make_float2(wx - 1, wy));
  const float dBotLeft  = dot(gBotLeft,  make_float2(wx,     wy - 1));
  const float dBotRight = dot(gBotRight, make_float2(wx - 1, wy - 1));

  // Calculate the smoothed distance between given point and the top-left corner
  const float tx = smooth(wx), ty = smooth(wy);

  // Interpolate with the other corners
  const float leftInterp  = lerp(dTopLeft, dBotLeft, ty);
  const float rightInterp = lerp(dTopRight, dBotRight, ty);

  return (lerp(leftInterp, rightInterp, tx) + 1.0) * 0.5;
}

/* Calculates several octaves of Perlin noise at coordinate (x, y)
 * (according to the given `params`), sums them together and returns the result.
 */
__device__ float sumOctaves(float x, float y, NoiseParams params) {
  float frequency = 1;
  float sum = noiseAt(x * frequency , y * frequency, params.seed);
  float amplitude = 1;
  float range = 1;
  for (int i = 1; i < params.octaves; i++) {
    frequency *= params.lacunarity;
    amplitude *= params.persistence;
    range += amplitude;
    sum += amplitude * noiseAt(x * frequency, y * frequency, params.seed);
  }
  return sum / range;
}

__global__ void perlin(int yStart, int height, NoiseParams params, uint8_t *outPixels) {
  // Pixel coordinates
  const auto px = CUID(x);
  const auto py = CUID(y) + yStart;

  if (px >= WIN_WIDTH || py >= yStart + height) return;

  auto noise = sumOctaves(px / params.ppu, py / params.ppu, params);

  // Convert noise to pixel
  const auto baseIdx = 4 * LIN(px, py, WIN_WIDTH);

  const auto val = noise * 255;

  outPixels[baseIdx + 0] = val;
  outPixels[baseIdx + 1] = val;
  outPixels[baseIdx + 2] = val;
  outPixels[baseIdx + 3] = 255;
}

void Perlin::calculate(uint8_t *hPixels, NoiseParams params, cudaStream_t *streams, int nStreams) {

  const auto partialHeight = WIN_HEIGHT / nStreams;
  const dim3 threads(16, 16);
  const dim3 blocks(std::ceil(WIN_WIDTH / 16.0), std::ceil(partialHeight / 16.0));

  std::cout << "Each stream will calculate " << partialHeight * WIN_WIDTH << " pixels." << std::endl;

  uint8_t *dPixels;
  CHECK(cudaMalloc((void**)&dPixels, sizeof(uint8_t) * 4 * WIN_WIDTH * WIN_HEIGHT));

  std::cout << "For each stream, ";
  std::cout << "block size = " << threads << ", grid size = " << blocks
            << ", total threads = " << threads.x * blocks.x * threads.y * blocks.y << std::endl;

  CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  // Note that all kernels use the same device pointer, but they all write on
  // different parts of the pointer memory, so no data race occurs.
  for (int i = 0; i < nStreams; ++i) {
    perlin<<<blocks, threads, 0, streams[i]>>>(partialHeight * i, partialHeight, params, dPixels);
  }

  CHECK(cudaDeviceSynchronize());

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Total kernel execution time " <<  time * 1e-6  << " (ms)" << std::endl;

  CHECK(cudaMemcpy(hPixels, dPixels, sizeof(uint8_t) * 4 * WIN_WIDTH * WIN_HEIGHT,
        cudaMemcpyDeviceToHost));

  CHECK(cudaFree(dPixels));
}
