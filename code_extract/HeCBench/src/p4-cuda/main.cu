/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "params.h"

__global__
void postprocess (
  const float *__restrict__ cls_input,
        float *__restrict__ box_input,
  const float *__restrict__ dir_cls_input,
  const float *__restrict__ anchors,
  const float *__restrict__ anchor_bottom_heights,
        float *__restrict__ bndbox_output,
        int *__restrict__ object_counter,
  const float min_x_range,
  const float max_x_range,
  const float min_y_range,
  const float max_y_range,
  const int feature_x_size,
  const int feature_y_size,
  const int num_anchors,
  const int num_classes,
  const int num_box_values,
  const float score_thresh,
  const float dir_offset)
{
  int loc_index = blockIdx.x;
  int ith_anchor = threadIdx.x;
  if (ith_anchor >= num_anchors) return;

  int col = loc_index % feature_x_size;
  int row = loc_index / feature_x_size;
  float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
  float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);
  int cls_offset = loc_index * num_anchors * num_classes + ith_anchor * num_classes;
  float dev_cls[2] = {-1.f, 0.f};

  const float *scores = cls_input + cls_offset;
  float max_score = 1.f / (1.f + expf(-scores[0]));
  int cls_id = 0;
  for (int i = 1; i < num_classes; i++) {
    float cls_score = 1.f / (1.f + expf(-scores[i]));
    if (cls_score > max_score) {
      max_score = cls_score;
      cls_id = i;
    }
  }
  dev_cls[0] = static_cast<float>(cls_id);
  dev_cls[1] = max_score;

  if (dev_cls[1] >= score_thresh)
  {
    const int box_offset = loc_index * num_anchors * num_box_values + ith_anchor * num_box_values;
    const int dir_cls_offset = loc_index * num_anchors * 2 + ith_anchor * 2;
    const float *anchor_ptr = anchors + ith_anchor * 4;
    const float z_offset = anchor_ptr[2] / 2 + anchor_bottom_heights[ith_anchor / 2];
    const float anchor[7] = {x_offset, y_offset, z_offset, anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};
    float *box_encodings = box_input + box_offset;

    const float xa = anchor[0];
    const float ya = anchor[1];
    const float za = anchor[2];
    const float dxa = anchor[3];
    const float dya = anchor[4];
    const float dza = anchor[5];
    const float ra = anchor[6];
    const float diagonal = sqrtf(dxa * dxa + dya * dya);
    box_encodings[0] = box_encodings[0] * diagonal + xa;
    box_encodings[1] = box_encodings[1] * diagonal + ya;
    box_encodings[2] = box_encodings[2] * dza + za;
    box_encodings[3] = expf(box_encodings[3]) * dxa;
    box_encodings[4] = expf(box_encodings[4]) * dya;
    box_encodings[5] = expf(box_encodings[5]) * dza;
    box_encodings[6] = box_encodings[6] + ra;

    const int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1] ? 0 : 1;
    const float period = (float)M_PI;
    const float val = box_input[box_offset + 6] - dir_offset;
    const float dir_rot = val - floorf(val / (period + 1e-8f)) * period;
    const float yaw = dir_rot + dir_offset + period * dir_label;

    int resCount = atomicAdd(object_counter, 1);
    bndbox_output[0] = resCount+1;
    float *data = bndbox_output + 1 + resCount * 9;
    data[0] = box_input[box_offset];
    data[1] = box_input[box_offset + 1];
    data[2] = box_input[box_offset + 2];
    data[3] = box_input[box_offset + 3];
    data[4] = box_input[box_offset + 4];
    data[5] = box_input[box_offset + 5];
    data[6] = yaw;
    data[7] = dev_cls[0];
    data[8] = dev_cls[1];
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Params p;  // constant values defined in params.h
  const float min_x_range = p.min_x_range;
  const float max_x_range = p.max_x_range;
  const float min_y_range = p.min_y_range;
  const float max_y_range = p.max_y_range;
  const int feature_x_size = p.feature_x_size;
  const int feature_y_size = p.feature_y_size;
  const int num_anchors = p.num_anchors;
  const int num_classes = p.num_classes;
  const int num_box_values = p.num_box_values;
  const float score_thresh = p.score_thresh;
  const float dir_offset = p.dir_offset;
  const int len_per_anchor = p.len_per_anchor;
  const int num_dir_bins = p.num_dir_bins;
  
  const int feature_size = feature_x_size * feature_y_size;
  const int feature_anchor_size = feature_size * num_anchors;
  const int cls_size = feature_anchor_size * num_classes;
  const int box_size = feature_anchor_size * num_box_values;
  const int dir_cls_size = feature_anchor_size * num_dir_bins;
  const int bndbox_size = feature_anchor_size * 9 + 1;

  const int cls_size_byte = cls_size * sizeof(float);
  const int box_size_byte = box_size * sizeof(float);
  const int dir_cls_size_byte = dir_cls_size * sizeof(float);
  const int bndbox_size_byte = bndbox_size * sizeof(float);

  // input of the post-process kernel
  float *h_cls_input = (float*) malloc (cls_size_byte);
  float *h_box_input = (float*) malloc (box_size_byte);
  float *h_dir_cls_input = (float*) malloc (dir_cls_size_byte);

  // output of the post-process kernel
  float *h_bndbox_output = (float*) malloc (bndbox_size_byte);

  // random values
  srand(123);
  for (int i = 0; i < cls_size; i++)  h_cls_input[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < box_size; i++)  h_box_input[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < dir_cls_size; i++)  h_dir_cls_input[i] = rand() / (float)RAND_MAX;
  
  float *d_cls_input, *d_box_input, *d_dir_cls_input, *d_bndbox_output;
  float *d_anchors, *d_anchor_bottom_heights;
  int *d_object_counter;

  cudaMalloc((void **)&d_cls_input, cls_size_byte);
  cudaMalloc((void **)&d_box_input, box_size_byte);
  cudaMalloc((void **)&d_dir_cls_input, dir_cls_size_byte);
  cudaMalloc((void **)&d_bndbox_output, bndbox_size_byte);

  cudaMemcpy(d_cls_input, h_cls_input, cls_size_byte, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dir_cls_input, h_dir_cls_input, dir_cls_size_byte, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_anchors, num_anchors * len_per_anchor * sizeof(float));
  cudaMalloc((void **)&d_anchor_bottom_heights, num_classes * sizeof(float));
  cudaMalloc((void **)&d_object_counter, sizeof(int));

  cudaMemcpy(d_anchors, p.anchors,
             num_anchors * len_per_anchor * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_anchor_bottom_heights, p.anchor_bottom_heights,
             num_classes * sizeof(float), cudaMemcpyHostToDevice);

  double time = 0.0;

  dim3 threads (num_anchors);
  dim3 blocks (feature_size);

  for (int i = 0; i < repeat; i++) {
    cudaMemcpy(d_box_input, h_box_input, box_size_byte, cudaMemcpyHostToDevice);
    cudaMemset(d_object_counter, 0, sizeof(int));

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    postprocess<<<blocks, threads>>> (
      d_cls_input,
      d_box_input,
      d_dir_cls_input,
      d_anchors,
      d_anchor_bottom_heights,
      d_bndbox_output,
      d_object_counter,
      min_x_range,
      max_x_range,
      min_y_range,
      max_y_range,
      feature_x_size,
      feature_y_size,
      num_anchors,
      num_classes,
      num_box_values,
      score_thresh,
      dir_offset);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average execution time of postprocess kernel: %f (us)\n", (time * 1e-3f) / repeat);

  cudaMemcpy(h_bndbox_output, d_bndbox_output, bndbox_size_byte, cudaMemcpyDeviceToHost);
  
  double checksum = 0.0;
  for (int i = 0; i < bndbox_size; i++) checksum += h_bndbox_output[i];
  printf("checksum = %lf\n", checksum / bndbox_size);

  cudaFree(d_anchors);
  cudaFree(d_anchor_bottom_heights);
  cudaFree(d_object_counter);
  cudaFree(d_cls_input);
  cudaFree(d_box_input);
  cudaFree(d_dir_cls_input);
  cudaFree(d_bndbox_output);

  free(h_cls_input);
  free(h_box_input);
  free(h_dir_cls_input);
  free(h_bndbox_output);

  return 0;
}
