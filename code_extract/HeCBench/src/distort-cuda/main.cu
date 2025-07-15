#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "distort.h"

int main(int argc, char **argv)
{
  if (argc != 5) {
    std::cout << "Usage: " << argv[0] <<
      "<input image width> <input image height> <coefficient of distortion> <repeat>\n";
    return 1;
  }

  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const float K = atof(argv[3]);
  const int repeat = atoi(argv[4]);

  struct Properties prop;
  prop.K = K;
  prop.centerX = width / 2;
  prop.centerY = height / 2;
  prop.width = width;
  prop.height = height;
  prop.thresh = 1.f;

  prop.xshift = calc_shift(0, prop.centerX - 1, prop.centerX, prop.K, prop.thresh);
  float newcenterX = prop.width - prop.centerX;
  float xshift_2 = calc_shift(0, newcenterX - 1, newcenterX, prop.K, prop.thresh);

  prop.yshift = calc_shift(0, prop.centerY - 1, prop.centerY, prop.K, prop.thresh);
  float newcenterY = prop.height - prop.centerY;
  float yshift_2 = calc_shift(0, newcenterY - 1, newcenterY, prop.K, prop.thresh);

  prop.xscale = (prop.width - prop.xshift - xshift_2) / prop.width;
  prop.yscale = (prop.height - prop.yshift - yshift_2) / prop.height;

  const int imageSize = height * width;
  const size_t imageSize_bytes = imageSize * sizeof(uchar3);

  uchar3* h_src = (uchar3*) malloc (imageSize_bytes);
  uchar3* h_dst = (uchar3*) malloc (imageSize_bytes);
  uchar3* r_dst = (uchar3*) malloc (imageSize_bytes);

  srand(123);
  for (int i = 0; i < imageSize; i++) {
    h_src[i] = make_uchar3(rand() % 256, rand() % 256, rand() % 256);
  }

  uchar3 *d_src, *d_dst;
  cudaMalloc((void**)&d_src, imageSize_bytes);
  cudaMemcpy(d_src, h_src, imageSize_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_dst, imageSize_bytes);

  struct Properties* d_prop;
  cudaMalloc((void**)&d_prop, sizeof(Properties));
  cudaMemcpy(d_prop, &prop, sizeof(Properties), cudaMemcpyHostToDevice);

  const dim3 blocks (16, 16);
  const dim3 grids (width / blocks.x + 1, height / blocks.y + 1);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    barrel_distort<<<grids, blocks>>>(d_src, d_dst, d_prop);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(h_dst, d_dst, imageSize_bytes, cudaMemcpyDeviceToHost);

  // verify
  int ex = 0, ey = 0, ez = 0;
  reference(h_src, r_dst, &prop);
  for (int i = 0; i < imageSize; i++) {
    ex = max(abs(h_dst[i].x - r_dst[i].x), ex);
    ey = max(abs(h_dst[i].y - r_dst[i].y), ey);
    ez = max(abs(h_dst[i].z - r_dst[i].z), ez);
  }

  std::cout << "Max error of each channel: " << ex << " " << ey << " " << ez << std::endl;

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_prop);

  free(h_src);
  free(h_dst);

  return 0;
}
