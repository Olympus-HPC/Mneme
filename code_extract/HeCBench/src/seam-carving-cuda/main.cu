#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <chrono>
#include <utility>  // std::swap
#include <cuda.h>
#include "utils.h"
#include "kernels.h"
#include "kernels_wrapper.h"

//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char **argv) {
  if(argc < 3){
    printf("Usage: %s <file> <number of seams to remove> [options]\n"
        "valid options:\n-u\tupdate costs instead of recomputing them\n"
        "-a\tapproximate computation\n", argv[0]);
    return 1;
  }

  char *check;
  long seams_to_remove = strtol(argv[2], &check, 10);  //10 specifies base-10
  if (check == argv[2]){   //if no characters were converted pointers are equal
    printf("ERROR: can't convert string to number, exiting.\n");
    return 1;
  }

  int w, h, ncomp;
  unsigned char* imgv = stbi_load(argv[1], &w, &h, &ncomp, 0);
  if(imgv == NULL){
    printf("ERROR: can't load image \"%s\" (maybe the file does not exist?), exiting.\n", argv[1]);
    return 1;
  }

  if(ncomp != 3){
    printf("ERROR: image does not have 3 components (RGB), exiting.\n");
    return 1;
  }

  if(seams_to_remove < 0 || seams_to_remove >= w){
    printf("ERROR: number of seams to remove is invalid, exiting.\n");
    return 1;
  }

  seam_carver_mode mode = SEAM_CARVER_STANDARD_MODE;

  if(argc >= 4){
    if(strcmp(argv[3],"-u") == 0){
      mode = SEAM_CARVER_UPDATE_MODE;
      printf("update mode selected.\n");
    }
    else if(strcmp(argv[3],"-a") == 0){
      mode = SEAM_CARVER_APPROX_MODE;
      printf("approximation mode selected.\n");
    }
    else{    
      printf("an invalid option was specified and will be ignored. Valid options are: -u, -a.\n");
    }
  }

  printf("Image loaded. Resizing...\n");

  int current_w = w;
  uchar4 *h_pixels = build_pixels(imgv, w, h);
  const int img_bytes = w * h * sizeof(uchar4);
  const int cost_bytes = w * h * sizeof(short);
  const int h_bytes = h * sizeof(int);
  const int w_bytes = w * sizeof(int);

  uchar4 *d_pixels;
  uchar4 *d_pixels_swap;
  short *d_costs_left, *d_costs_swap_left;
  short *d_costs_up, *d_costs_swap_up;
  short *d_costs_right, *d_costs_swap_right;
  int *d_index_map;
  int *d_offset_map;
  int *d_indices_ref;
  int *d_indices;
  int *d_seam;
  int *reduce_row; //M row to consider for reduce
  int *d_M;

  if(mode != SEAM_CARVER_APPROX_MODE) {
    cudaMalloc((void**)&d_costs_left, cost_bytes); 
    cudaMalloc((void**)&d_costs_up, cost_bytes); 
    cudaMalloc((void**)&d_costs_right, cost_bytes); 
  }
  if(mode == SEAM_CARVER_UPDATE_MODE){
    cudaMalloc((void**)&d_costs_swap_left, cost_bytes); 
    cudaMalloc((void**)&d_costs_swap_up, cost_bytes); 
    cudaMalloc((void**)&d_costs_swap_right, cost_bytes);
  }
  //sum map in approx mode
  cudaMalloc((void**)&d_M, img_bytes); 

  // rows to consider for reduce
  if(mode == SEAM_CARVER_APPROX_MODE)
    reduce_row = d_M; //first row
  else
    reduce_row = d_M + w*(h-1); //last row

  if(mode == SEAM_CARVER_APPROX_MODE){
    cudaMalloc((void**)&d_index_map, img_bytes);
    cudaMalloc((void**)&d_offset_map, img_bytes);
  }

  int* indices = (int*)malloc(w_bytes);
  for(int i = 0; i < w; i++) indices[i] = i;

  cudaMalloc((void**)&d_indices, w_bytes); 

  cudaMalloc((void**)&d_indices_ref, w_bytes); 
  cudaMemcpy(d_indices_ref, indices, w_bytes, cudaMemcpyHostToDevice);   
  cudaMalloc((void**)&d_seam, h_bytes);    

  cudaMalloc((void**)&d_pixels, img_bytes);
  cudaMemcpy(d_pixels, h_pixels, img_bytes, cudaMemcpyHostToDevice);       
  cudaMalloc((void**)&d_pixels_swap, img_bytes);

  if(mode == SEAM_CARVER_UPDATE_MODE)
    compute_costs(current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

  int num_iterations = 0;

  auto start = std::chrono::steady_clock::now();

  while(num_iterations < (int)seams_to_remove){

    if(mode == SEAM_CARVER_STANDARD_MODE)
      compute_costs(current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

    if(mode != SEAM_CARVER_APPROX_MODE){
      compute_M(current_w, w, h, d_M, d_costs_left, d_costs_up, d_costs_right);
      find_min_index(current_w, d_indices_ref, d_indices, reduce_row);
      find_seam(current_w, w, h, d_M, d_indices, d_seam);
    }
    else{
      approx_setup(current_w, w, h, d_pixels, d_index_map, d_offset_map, d_M);
      approx_M(current_w, w, h, d_offset_map,  d_M);
      find_min_index(current_w, d_indices_ref, d_indices, reduce_row);
      approx_seam(w, h, d_index_map, d_indices, d_seam);
    }

    remove_seam(current_w, w, h, d_M, d_pixels, d_pixels_swap, d_seam);
    std::swap(d_pixels, d_pixels_swap);

    if(mode == SEAM_CARVER_UPDATE_MODE){
      update_costs(current_w, w, h, d_M, d_pixels, 
                   d_costs_left, d_costs_up, d_costs_right,
                   d_costs_swap_left, d_costs_swap_up, d_costs_swap_right, d_seam );
      std::swap(d_costs_left, d_costs_swap_left);
      std::swap(d_costs_up, d_costs_swap_up);
      std::swap(d_costs_right, d_costs_swap_right);
    }

    current_w--;
    num_iterations++;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  float time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of seam carver kernels: %f (ms)\n", time * 1e-6f);

  cudaMemcpy(h_pixels, d_pixels, img_bytes, cudaMemcpyDeviceToHost);
  unsigned char* output = flatten_pixels(h_pixels, w, h, current_w); 
  printf("Image resized\n");

  printf("Saving in resized.bmp...\n");
  int success = stbi_write_bmp("resized.bmp", current_w, h, 3, output);
  printf("%s\n", success ? "Success" : "Failed");

  cudaFree(d_pixels);
  cudaFree(d_pixels_swap);
  if(mode != SEAM_CARVER_APPROX_MODE){
    cudaFree(d_costs_left);
    cudaFree(d_costs_up);
    cudaFree(d_costs_right);
  }
  if(mode == SEAM_CARVER_UPDATE_MODE){
    cudaFree(d_costs_swap_left);
    cudaFree(d_costs_swap_up);
    cudaFree(d_costs_swap_right);
  }
  cudaFree(d_M); 
  cudaFree(d_indices); 
  cudaFree(d_indices_ref); 
  cudaFree(d_seam);
  if(mode == SEAM_CARVER_APPROX_MODE){
    cudaFree(d_index_map);
    cudaFree(d_offset_map);
  }
  free(h_pixels);
  free(output);   
  free(indices);
  return 0;
}
