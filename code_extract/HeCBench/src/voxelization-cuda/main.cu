/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <chrono>
#include <dirent.h>
#include "kernels.h"

void generateVoxels(const float *points, size_t points_size, const int repeat)
{
  unsigned int hash_table_size_;
  unsigned int voxels_temp_size_;
  unsigned int voxel_features_size_;
  unsigned int voxel_idxs_size_;
  unsigned int voxel_num_size_;

  hash_table_size_ = MAX_POINTS_NUM * 2 * 2 * sizeof(unsigned int);

  voxels_temp_size_ =
    params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(float);
  voxel_features_size_ =
    params_.max_voxels * params_.max_points_per_voxel * params_.feature_num * sizeof(half);
  voxel_num_size_ = params_.max_voxels * sizeof(unsigned int);
  voxel_idxs_size_ = params_.max_voxels * 4 * sizeof(unsigned int);

  unsigned int *hash_table_;
  float *voxels_temp_;

  unsigned int *d_real_num_voxels_;
  unsigned int *h_real_num_voxels_;

  half *d_voxel_features_;
  unsigned int *d_voxel_num_;
  unsigned int *d_voxel_indices_;

  checkCudaErrors(cudaMalloc((void **)&d_voxel_features_, voxel_features_size_));
  checkCudaErrors(cudaMalloc((void **)&d_voxel_num_, voxel_num_size_));
  checkCudaErrors(cudaMalloc((void **)&d_voxel_indices_, voxel_idxs_size_));
  checkCudaErrors(cudaMalloc((void **)&d_real_num_voxels_, sizeof(unsigned int)));

  checkCudaErrors(cudaMemset(d_voxel_num_, 0, voxel_num_size_));
  checkCudaErrors(cudaMemset(d_voxel_features_, 0, voxel_features_size_));
  checkCudaErrors(cudaMemset(d_voxel_indices_, 0, voxel_idxs_size_));
  checkCudaErrors(cudaMemset(d_real_num_voxels_, 0, sizeof(unsigned int)));

  checkCudaErrors(cudaMalloc((void **)&hash_table_, hash_table_size_));
  checkCudaErrors(cudaMalloc((void **)&voxels_temp_, voxels_temp_size_));
  checkCudaErrors(cudaMemset(hash_table_, 0xff, hash_table_size_));
  checkCudaErrors(cudaMemset(voxels_temp_, 0xff, voxels_temp_size_));

  checkCudaErrors(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    checkCudaErrors(voxelizationLaunch(points, points_size,
          params_.min_x_range, params_.max_x_range,
          params_.min_y_range, params_.max_y_range,
          params_.min_z_range, params_.max_z_range,
          params_.pillar_x_size, params_.pillar_y_size, params_.pillar_z_size,
          params_.getGridXSize(), params_.getGridYSize(), params_.getGridZSize(),
          params_.feature_num, params_.max_voxels,
          params_.max_points_per_voxel, hash_table_,
          d_voxel_num_, voxels_temp_, d_voxel_indices_,
          d_real_num_voxels_));
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the voxelization kernel: %f (us)\n", (time * 1e-3f) / repeat);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMallocHost((void **)&h_real_num_voxels_, sizeof(unsigned int)));
  checkCudaErrors(cudaMemcpy(h_real_num_voxels_, d_real_num_voxels_, sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "valid_num: " << *h_real_num_voxels_ <<std::endl;

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    checkCudaErrors(featureExtractionLaunch(voxels_temp_, d_voxel_num_,
          h_real_num_voxels_, params_.max_points_per_voxel, params_.feature_num,
          d_voxel_features_));
  }

  checkCudaErrors(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the feature extraction kernel: %f (us)\n", (time * 1e-3f) / repeat);

  checkCudaErrors(cudaFree(hash_table_));
  checkCudaErrors(cudaFree(voxels_temp_));
  checkCudaErrors(cudaFree(d_voxel_features_));
  checkCudaErrors(cudaFree(d_voxel_num_));
  checkCudaErrors(cudaFree(d_voxel_indices_));
  checkCudaErrors(cudaFree(d_real_num_voxels_));
  checkCudaErrors(cudaFreeHost(h_real_num_voxels_));
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(path)) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string file = ent->d_name;
      if(hasEnding(file, suffix)){
        files.push_back(file.substr(0, file.length()-4));
      }
    }
    closedir(dir);
  } else {
    std::cout << "No such folder: " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open()) {
    std::cout << "Can't open files: "<< file<<std::endl;
    return -1;
  }

  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  char *buffer = new char[len];
  if (buffer==NULL) {
    std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
    std::exit(EXIT_FAILURE);
  }

  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

static void help()
{
  std::cout << "Usage: \n    ./main data/test/  100\n";
  std::cout << "    Run voxelization with data under ./data/test/ for 100 times\n";
  std::exit(EXIT_SUCCESS);
}

int main(int argc, const char **argv)
{
  if (argc < 3)
    help();

  const char *data_folder  = argv[1];
  const int repeat  = atoi(argv[2]);

  std::vector<std::string> files;
  getFolderFile(data_folder, files);

  std::cout << "Number of files: " << files.size() << std::endl;

  Params params;

  float *d_points = nullptr;    
  checkCudaErrors(cudaMalloc((void **)&d_points, MAX_POINTS_NUM * params.feature_num * sizeof(float)));
  for (const auto & file : files)
  {
    std::string dataFile = data_folder + file + ".bin";

    std::cout << "\n<<<<<<<<<<<" <<std::endl;
    std::cout << "load file: "<< dataFile <<std::endl;

    unsigned int length = 0;
    void *pc_data = NULL;

    loadData(dataFile.c_str() , &pc_data, &length);
    size_t points_num = length / (params.feature_num * sizeof(float)) ;
    std::cout << "find points num: " << points_num << std::endl;

    checkCudaErrors(cudaMemcpy(d_points, pc_data, length, cudaMemcpyHostToDevice));

    generateVoxels(d_points, points_num, repeat);

    std::cout << ">>>>>>>>>>>" <<std::endl;
  }

  checkCudaErrors(cudaFree(d_points));
  return 0;
}
