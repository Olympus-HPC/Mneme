#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>

__global__
void zoom_in_kernel(
    const float *input_tensor,float *output_tensor,
    int input_h, int input_w, int output_h, int output_w,
    int pitch, int out_h_start, int out_h_end,
    int out_w_start, int out_w_end)
{
  extern __shared__ float staging_tile[];

  // H -> block Y, row
  // W -> block X, col
  int out_start_h = blockIdx.y * blockDim.y;
  int out_end_h   = (blockIdx.y + 1) * blockDim.y - 1;
  int out_start_w = blockIdx.x * blockDim.x;
  int out_end_w   = (blockIdx.x + 1) * blockDim.x - 1;

  int img_start_offset = blockIdx.z * pitch;

  // ideally should go in unified register
  int smem_load_h_start = floorf((out_start_h * input_h) / (float)output_h);
  int smem_load_h_end = ceilf(((out_end_h+1) * input_h) / (float)output_h);
  int smem_h_load_stretch = smem_load_h_end - smem_load_h_start;

  int smem_load_w_start = floorf((out_start_w * input_w) / (float)output_w);
  int smem_load_w_end = ceilf(((out_end_w+1) * input_w) / (float)output_w);
  int smem_w_load_stretch = smem_load_w_end - smem_load_w_start;

  for (int i = threadIdx.y; i < smem_h_load_stretch; i+=blockDim.y) {
    for (int j = threadIdx.x; j < smem_w_load_stretch; j+=blockDim.x) {

      if (((i+smem_load_h_start) < input_h) &&
          ((j+smem_load_w_start) < input_w)) {
        staging_tile[i * smem_w_load_stretch + j] = \
                      input_tensor[img_start_offset +
                      (smem_load_h_start + i) * input_w +
                      smem_load_w_start + j];
      } else {
        staging_tile[i * smem_w_load_stretch + j] = 0.0f;
      }
    }
  }
  __syncthreads();

  int out_pixel_h = blockIdx.y * blockDim.y + threadIdx.y;
  int out_pixel_w = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_pixel_h < output_h && out_pixel_w < output_w
      && out_pixel_h >= out_h_start && out_pixel_h < out_h_end
      && out_pixel_w >= out_w_start && out_pixel_w < out_w_end) {

    // compute pixels oh, ow span
    int start_h = floorf((out_pixel_h * input_h) / (float)output_h);
    int end_h = ceilf(((out_pixel_h+1) * input_h) / (float)output_h);

    int start_w = floorf((out_pixel_w * input_w) / (float)output_w);
    int end_w = ceilf(((out_pixel_w+1) * input_w) / (float)output_w);

    int del_h = end_h - start_h;
    int del_w = end_w - start_w;

    float sum_ = 0.0f;

    for (int i = 0; i < del_h; i++) {
      for (int j = 0; j < del_w; j++) {
        int smem_row = (start_h + i) - smem_load_h_start;
        int smem_col = (start_w + j) - smem_load_w_start;
        sum_ += staging_tile[smem_row * smem_w_load_stretch + smem_col];
      }
    }
    sum_ /= (float)del_h;
    sum_ /= (float)del_w;

    output_tensor[(blockIdx.z * pitch) +
      ((out_pixel_h - out_h_start) * input_w) +
      (out_pixel_w - out_w_start)] = sum_;
  }
}

__global__
void zoom_out_kernel(
    const float *input_tensor, float *output_tensor,
    int input_h, int input_w, int output_h, int output_w,
    int pitch, int out_h_start, int out_h_end, int out_w_start,
    int out_w_end)
{
  extern __shared__ float staging_tile[];

  // H -> block Y, row
  // W -> block X, col
  int out_start_h = blockIdx.y * blockDim.y;
  int out_end_h   = (blockIdx.y + 1) * blockDim.y - 1;
  int out_start_w = blockIdx.x * blockDim.x;
  int out_end_w   = (blockIdx.x + 1) * blockDim.x - 1;

  int img_start_offset = blockIdx.z * pitch;

  // ideally should go in unified register
  int smem_load_h_start = floorf((out_start_h * input_h) / (float)output_h);
  int smem_load_h_end = ceilf(((out_end_h+1) * input_h) / (float)output_h);
  int smem_h_load_stretch = smem_load_h_end - smem_load_h_start;

  int smem_load_w_start = floorf((out_start_w * input_w) / (float)output_w);
  int smem_load_w_end = ceilf(((out_end_w+1) * input_w) / (float)output_w);
  int smem_w_load_stretch = smem_load_w_end - smem_load_w_start;

  for (int i = threadIdx.y; i < smem_h_load_stretch; i+=blockDim.y) {
    for (int j = threadIdx.x; j < smem_w_load_stretch; j+=blockDim.x) {

      if (((i+smem_load_h_start) < input_h) &&
          ((j+smem_load_w_start) < input_w)) {
        staging_tile[i * smem_w_load_stretch + j] = \
                      input_tensor[img_start_offset +
                      (smem_load_h_start + i)*input_w +
                      smem_load_w_start + j];
      } else {
        staging_tile[i * smem_w_load_stretch + j] = 0.0f;
      }
    }
  }
  __syncthreads();

  int out_pixel_h = blockIdx.y * blockDim.y + threadIdx.y;
  int out_pixel_w = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_pixel_h < output_h && out_pixel_w < output_w) {

    // compute pixels oh, ow span
    int start_h = floorf((out_pixel_h * input_h) / (float)output_h);
    int end_h = ceilf(((out_pixel_h+1) * input_h) / (float)output_h);

    int start_w = floorf((out_pixel_w * input_w) / (float)output_w);
    int end_w = ceilf(((out_pixel_w+1) * input_w) / (float)output_w);

    int del_h = end_h - start_h;
    int del_w = end_w - start_w;

    float sum_ = 0.0f;

    for (int i = 0; i < del_h; i++) {
      for (int j = 0; j < del_w; j++) {
        int smem_row = (start_h + i) - smem_load_h_start;
        int smem_col = (start_w + j) - smem_load_w_start;
        sum_ += staging_tile[smem_row * smem_w_load_stretch + smem_col];
      }
    }
    sum_ /= (float)del_h;
    sum_ /= (float)del_w;

    output_tensor[(blockIdx.z * pitch) +
      ((out_pixel_h + out_h_start) * input_w) +
      (out_pixel_w + out_w_start)] = sum_;
  }
}

__global__
void zoom_out_edge_pad(
    float *output_tensor,
    int height, int width,
    int pitch, int no_padding_h_start,
    int no_padding_w_start,
    int no_padding_h_end, int no_padding_w_end) {
  // H -> block Y, row
  // W -> block X, col

  int out_pixel_h = blockIdx.y * blockDim.y + threadIdx.y;
  int out_pixel_w = blockIdx.x * blockDim.x + threadIdx.x;

  // no_padding_h_end, no_padding_w_end --> w_cropped+wstart, same for height
  int out_location = (blockIdx.z * pitch) + (out_pixel_h * width) + out_pixel_w;

  if (out_pixel_h < height && out_pixel_w < width) {
    if (out_pixel_h < no_padding_h_start && out_pixel_w >= no_padding_w_start
        && out_pixel_w < no_padding_w_end) {
      // top pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        (no_padding_h_start * width) + out_pixel_w];
    } else if (out_pixel_h >= no_padding_h_end
        && out_pixel_w >= no_padding_w_start
        && out_pixel_w < no_padding_w_end) {
      // bottom pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        ((no_padding_h_end-1) * width) + out_pixel_w];
    } else if (out_pixel_w < no_padding_w_start
        && out_pixel_h >= no_padding_h_start
        && out_pixel_h < no_padding_h_end) {
      // left pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        (out_pixel_h * width) + no_padding_w_start];
    } else if (out_pixel_w >= no_padding_w_end
        && out_pixel_h >= no_padding_h_start
        && out_pixel_h < no_padding_h_end) {
      // right pad
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        (out_pixel_h * width) + (no_padding_w_end-1)];
    } else if (out_pixel_h < no_padding_h_start
        && out_pixel_w < no_padding_w_start) {
      // top-left corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        (no_padding_h_start * width) +
        no_padding_w_start];
    } else if (out_pixel_h < no_padding_h_start
        && out_pixel_w >= no_padding_w_end) {
      // top-right corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        (no_padding_h_start * width) +
        (no_padding_w_end-1)];
    } else if (out_pixel_h >= no_padding_h_end
        && out_pixel_w < no_padding_w_start) {
      // bottom-left corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        ((no_padding_h_end-1) * width) +
        no_padding_w_start];
    } else if (out_pixel_h >= no_padding_h_end
        && out_pixel_w >= no_padding_w_end) {
      // bottom-right corner
      output_tensor[out_location] = output_tensor[(blockIdx.z * pitch) +
        ((no_padding_h_end-1) * width) +
        (no_padding_w_end-1)];
    }
  }
}

int get_sm_size(int *output_sizes, int H, int W, dim3 &block) {

  int max_smem = 48 * 1024;
  int blocks[4][3] = {{16, 16, 1}, {16, 8, 1}, {8, 8, 1}, {8, 4, 1}};
  for (int i = 0; i < 4; i++) {
    int *param = blocks[0];
    int h_stretch = ceil((param[1] * H) / (float)output_sizes[2]);
    int w_stretch = ceil((param[0] * W) / (float)output_sizes[3]);
    int smem_size = (h_stretch + 1) * (w_stretch + 1) * 4;
    if (smem_size < max_smem) {
      block.x = param[0];
      block.y = param[1];
      block.z = param[2];
      return smem_size;
    }
  }
  printf("Requested shared memory size exceeds the maximum (%d). Exit\n", max_smem);
  exit(1);
}

void zoom (int repeat, int input_sizes[4], float zoom_factor[2])
{
  int N = input_sizes[0];
  int C = input_sizes[1];
  int H = input_sizes[2];
  int W = input_sizes[3];

  // {N, C, H, W} 
  int output_sizes[] = {N, C, (int)floor(H * zoom_factor[0]), (int)floor(W * zoom_factor[1])};

  bool is_zoom_out = output_sizes[2] < H && output_sizes[3] < W;
  bool is_zoom_in = output_sizes[2] > H && output_sizes[3] > W;
  if (is_zoom_out == false && is_zoom_in == false) {
    printf("Zoom factors only handle simultaneous expansion(or shrinkage) in both dimensions. Exit\n");
    exit(1);
  }

  // input pitch
  int pitch = H * W;

  // get block size and shared memory size
  dim3 block;
  int smem_size = get_sm_size(output_sizes, H, W, block);

  dim3 grid (int((output_sizes[3] - 1) / block.x + 1),
             int((output_sizes[2] - 1) / block.y + 1), C * N);

  int pad_dims[2][2] = {{0, 0}, {0,0}};  // zoom out
  int slice_dims[2][2] = {{0, 0}, {0,0}};  // zoom in

  int diff = H - output_sizes[2];
  int half = abs(diff) / 2;
  if (diff > 0) {
    pad_dims[0][0] = half;
    pad_dims[0][1] = diff - half;
  } else {
    slice_dims[0][0] = half;
    slice_dims[0][1] = H + half;
  }

  diff = W - output_sizes[3];
  half = abs(diff) / 2;
  if (diff > 0) {
    pad_dims[1][0] = half;
    pad_dims[1][1] = diff - half;
  } else {
    slice_dims[1][0] = half;
    slice_dims[1][1] = W + half;
  }

  size_t img_size = N * C * H * W; 
  size_t img_size_bytes = sizeof(float) * img_size;

  float *input_img = (float*) malloc (img_size_bytes);

  float *d_input_img;
  cudaMalloc((void**)&d_input_img, img_size_bytes);

  float *output_img = (float*) malloc (img_size_bytes);

  float *d_output_img;
  cudaMalloc((void**)&d_output_img, img_size_bytes);

  std::default_random_engine rng (123);
  std::normal_distribution<float> norm_dist(0.f, 1.f);

  for (size_t i = 0; i < img_size; i++) {
    input_img[i] = norm_dist(rng);
  }

  cudaMemcpy(d_input_img, input_img, img_size_bytes, cudaMemcpyHostToDevice);

  long total_time = 0;
  for (int i = 0; i < repeat; i++) {

    auto start = std::chrono::steady_clock::now();
    cudaMemset(d_output_img, 0, img_size_bytes);
    cudaDeviceSynchronize();

    if (is_zoom_in) {
      zoom_in_kernel <<<grid, block, smem_size>>> (
          d_input_img, d_output_img, H, W, 
          output_sizes[2],
          output_sizes[3],
          pitch, slice_dims[0][0],
          slice_dims[0][1],
          slice_dims[1][0],
          slice_dims[1][1]);
    }
    else if (is_zoom_out) {
      zoom_out_kernel <<<grid, block, smem_size>>> (
        d_input_img, d_output_img, H, W, 
        output_sizes[2],
        output_sizes[3],
        pitch, pad_dims[0][0],
        pad_dims[0][1],
        pad_dims[1][0],
        pad_dims[1][1]);

      dim3 grid2 (int((W - 1) / block.x + 1),
                  int((H - 1) / block.y + 1),
                  C * N);

      zoom_out_edge_pad <<<grid2, block>>> (
        d_output_img, H, W, pitch, 
        pad_dims[0][0], pad_dims[1][0],
        pad_dims[0][0] + output_sizes[2],
        pad_dims[1][0] + output_sizes[3]);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  cudaMemcpy(output_img, d_output_img, img_size_bytes, cudaMemcpyDeviceToHost);

  double checksum = 0;
  for (size_t i = 0; i < img_size; i++) {
    checksum += output_img[i];
  }

  cudaFree(d_input_img);
  cudaFree(d_output_img);
  free(input_img);
  free(output_img);

  printf("Average execution time of the %s kernel: %f (us)\n",
         is_zoom_in ? "zoom-in" : "zoom-out", total_time * 1e-3 / repeat);
  printf("Kernel checksum: %lf\n", checksum);
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <batch> <channel> <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  int input_sizes[4];
  input_sizes[0] = atoi(argv[1]);
  input_sizes[1] = atoi(argv[2]);
  input_sizes[2] = atoi(argv[3]);
  input_sizes[3] = atoi(argv[4]);
  int repeat = atoi(argv[5]);

  float zf[2]; // zoom factor
  
  zf[0] = 1.5f; zf[1] = 2.5f;
  zoom(repeat, input_sizes, zf);

  zf[0] = 0.6f; zf[1] = 0.9f;
  zoom(repeat, input_sizes, zf);

  return 0;
}
