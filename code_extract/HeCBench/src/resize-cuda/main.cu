#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

template <class T, std::size_t CHANNELS_PER_ITER>
__global__
void resize (
    T *__restrict__ output,
    size_t output_size, int out_height, int out_width,
    const T *__restrict__ input, int in_height, int in_width,
    float o2i_fy, float o2i_fx, bool round, bool half_pixel_centers)
{
    auto in_image_size = in_height * in_width;
    auto out_image_size = out_height * out_width;

    /* think of the output and input as a collection of 2d images with the last axis
     * representing the width and the last but one axis representing the height
     *
     * the remaining axis together form a collection of these images/channels
     */
    auto num_effective_channels = output_size / out_image_size;

    /* we process multiple channels every iteration to reuse the identical computation
     * involved with the spatial dimensions
     *
     * if we are processing `CHANNELS_PER_ITER` channels per iteration, we will need
     * (num_effective_channels / CHANNELS_PER_ITER) iterations per (x, y) location
     */
    auto num_channel_iters_per_xy = (num_effective_channels / CHANNELS_PER_ITER);

    /* we need `num_channel_iters_per_xy` iterations per (x, y) and there are `out_image_size`
     * combinations of (x, y); hence, we'll need `num_channel_iters_per_xy * out_image_size`
     * iterations in total to finish the resize operation
     */
    auto iters_required = num_channel_iters_per_xy * out_image_size;

    for (int iter = blockIdx.x * blockDim.x + threadIdx.x;
             iter < iters_required; iter += blockDim.x * gridDim.x) {

        const int c_start = (iter / out_image_size) * CHANNELS_PER_ITER;

        /* note here that consecutive `iter` values will often have consecutive `x` values
         * => stores into output will be coalesced across threads
         */
        const int y = (iter % out_image_size) / out_width;
        const int x = iter % out_width;

        auto in_yf = half_pixel_centers ? (y + 0.5f) * o2i_fy : y * o2i_fy;
        int in_y = round ? lroundf(in_yf) : static_cast<int>(in_yf);

        auto in_xf = half_pixel_centers ? (x + 0.5f) * o2i_fx : x * o2i_fx;
        int in_x = round ? lroundf(in_xf) : static_cast<int>(in_xf);

        in_x = min(in_x, in_width - 1);
        in_y = min(in_y, in_height - 1);

        int in_idx = c_start * in_image_size + in_y * in_width + in_x;
        int out_idx = c_start * out_image_size + y * out_width + x;

        for (int i = 0; i < CHANNELS_PER_ITER; i++) {
            output[out_idx] = input[in_idx];
            in_idx += in_image_size;
            out_idx += out_image_size;
        }
    }
}

template <class T, std::size_t CHANNELS_PER_ITER>
__global__ void resize_bilinear(
    T *__restrict__ output,
    size_t output_size, int out_height, int out_width,
    const T *__restrict__ input, int in_height, int in_width,
    float o2i_fy, float o2i_fx, bool half_pixel_centers)
{
    auto in_image_size = in_height * in_width;
    auto out_image_size = out_height * out_width;

    /* think of the output and input as a collection of 2d images with the last axis
     * representing the width and the last but one axis representing the height
     *
     * the remaining axis together form a collection of these images/channels
     */
    auto num_effective_channels = output_size / out_image_size;

    /* we process multiple channels every iteration to reuse the identical computation
     * involved with the spatial dimensions
     *
     * if we are processing `CHANNELS_PER_ITER` channels per iteration, we will need
     * (num_effective_channels / CHANNELS_PER_ITER) iterations per (x, y) location
     */
    auto num_channel_iters_per_xy = (num_effective_channels / CHANNELS_PER_ITER);

    /* we need `num_channel_iters_per_xy` iterations per (x, y) and there are `out_image_size`
     * combinations of (x, y); hence, we'll need `num_channel_iters_per_xy * out_image_size`
     * iterations in total to finish the resize operation
     */
    auto iters_required = num_channel_iters_per_xy * out_image_size;

    for (int iter = blockIdx.x * blockDim.x + threadIdx.x;
             iter < iters_required; iter += blockDim.x * gridDim.x) {

        const int c_start = (iter / out_image_size) * CHANNELS_PER_ITER;
        const int c_end = c_start + CHANNELS_PER_ITER;

        /* note here that consecutive `iter` values will often have consecutive `x` values
         * => stores into output will be coalesced across threads
         */
        const int y = (iter % out_image_size) / out_width;
        const int x = iter % out_width;

        auto in_x = half_pixel_centers ? fmaxf((x + 0.5f) * o2i_fx - 0.5f, 0.0f) : x * o2i_fx;
        auto in_y = half_pixel_centers ? fmaxf((y + 0.5f) * o2i_fy - 0.5f, 0.0f) : y * o2i_fy;

        auto in_x0 = static_cast<int>(in_x);
        auto in_x1 = min(in_x0 + 1, in_width - 1);

        auto in_y0 = static_cast<int>(in_y);

        auto in_y1 = min(in_y0, in_height - 1);
        auto in_y2 = min(in_y0 + 1, in_height - 1);

        int in_offset_r0 = c_start * in_image_size + in_y1 * in_width;
        int in_offset_r1 = c_start * in_image_size + in_y2 * in_width;
        int out_idx = c_start * out_image_size + y * out_width + x;

        #pragma unroll 1 /* disable unrolling to reduce register pressure; not sure how but it works */
        for (auto c = c_start; c < c_end; c++) {
            auto v_00 = input[in_offset_r0 + in_x0],
                 v_01 = input[in_offset_r0 + in_x1],
                 v_10 = input[in_offset_r1 + in_x0],
                 v_11 = input[in_offset_r1 + in_x1];

            output[out_idx] =
                v_00 +
                T(in_y - in_y0) * T(v_10 - v_00) +
                T(in_x - in_x0) * T(v_01 - v_00) +
                T(in_y - in_y0) * T(in_x - in_x0) * T(v_11 - v_01 - v_10 + v_00);

            in_offset_r0 += in_image_size;
            in_offset_r1 += in_image_size;
            out_idx += out_image_size;
        }
    }
}

template <class T>
void resize_image (
  const int in_width,
  const int in_height,
  const int out_width,
  const int out_height,
  const int num_channels,
  const int repeat,
  const bool bilinear = false)
{
  size_t in_image_size = (size_t)in_height * in_width;
  size_t in_size = num_channels * in_image_size;
  size_t in_size_bytes = sizeof(T) * in_size;

  size_t out_image_size = (size_t)out_height * out_width;
  size_t out_size = num_channels * out_image_size;
  size_t out_size_bytes = sizeof(T) * out_size;

  T* in_images_h = (T*) malloc (in_size_bytes);
  T* out_images_h = (T*) malloc (out_size_bytes);

  for(size_t i = 0; i < in_size; i++) in_images_h[i] = static_cast<T>((i+1) % 13);

  T *in_images_d, *out_images_d;
  cudaMalloc((void**)&in_images_d, in_size_bytes);
  cudaMemcpy(in_images_d, in_images_h, in_size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&out_images_d, out_size_bytes);
  cudaMemset(out_images_d, 0, out_size_bytes);

  const float fx = in_width / out_width;
  const float fy = in_height / out_height;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // default grid size is 256 * 114
  if (bilinear) {
    for (int i = 0; i < repeat; i++) {
      resize_bilinear<T, 8> <<<29184, 256>>> (
        out_images_d, out_size, out_height, out_width,
        in_images_d, in_height, in_width, fx, fy, true);
    }
  } else {
    for (int i = 0; i < repeat; i++) {
      resize<T, 8> <<<29184, 256>>> (
        out_images_d, out_size, out_height, out_width,
        in_images_d, in_height, in_width, fx, fy, true, true);
    }
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %lf (us)    Perf: %lf (GB/s)\n",
         time * 1e-3 / repeat, (in_size_bytes + out_size_bytes) * repeat * 1.0 / time);
  
  cudaMemcpy(out_images_h, out_images_d, out_size_bytes, cudaMemcpyDeviceToHost);

  cudaFree(in_images_d);
  cudaFree(out_images_d);

  free(in_images_h);
  free(out_images_h);
}

int main(int argc, char* argv[]) {
  if (argc != 7) {
    printf("Usage: %s <input image width> <input image height>\n", argv[0]);
    printf("          <output image width> <output image height>\n");
    printf("          <image channels> <repeat>\n");
    return 1;
  }

  const int in_width = atoi(argv[1]);
  const int in_height = atoi(argv[2]);
  const int out_width = atoi(argv[3]);
  const int out_height = atoi(argv[4]);
  const int num_channels = atoi(argv[5]);
  const int repeat = atoi(argv[6]);

  printf("Resize %d images from (%d x %d) to (%d x %d)\n",
          num_channels, in_width, in_height, out_width, out_height);

  printf("\nThe size of each pixel is 1 byte\n");
  resize_image<unsigned char>(in_width, in_height, out_width, out_height, num_channels, repeat);
  printf("\nBilinear resizing\n");
  resize_image<unsigned char>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

  printf("\nThe size of each pixel is 2 bytes\n");
  resize_image<unsigned short>(in_width, in_height, out_width, out_height, num_channels, repeat);
  printf("\nBilinear resizing\n");
  resize_image<unsigned short>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

  printf("\nThe size of each pixel is 4 bytes\n");
  resize_image<unsigned int>(in_width, in_height, out_width, out_height, num_channels, repeat);
  printf("\nBilinear resizing\n");
  resize_image<unsigned int>(in_width, in_height, out_width, out_height, num_channels, repeat, true);

  return 0;
}
