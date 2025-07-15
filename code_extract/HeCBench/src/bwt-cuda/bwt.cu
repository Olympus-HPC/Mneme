#include <iostream>
#include <list>
#include <cuda.h>
#include "bwt.hpp"

const int blockSize = 256;

__global__ void generate_table(int* table, int table_size, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < table_size; i+=stride)
    table[i] = (i < n) ? i : -1;
}

__device__ bool compare_rotations(const int& a, const int& b, const char* genome, int n) {
  if (a < 0) return false;
  if (b < 0) return true;
  for(int i = 0; i < n; i++) {
    if (genome[(a + i) % n] != genome[(b + i) % n]) {
      return genome[(a + i) % n] < genome[(b + i) % n];
    }
  }
  return false;
}

__global__ void bitonic_sort_step(int*__restrict table, int table_size, 
                                  int j, int k, const char*__restrict genome, int n) {
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int ixj = i ^ j;
  if (i < table_size && ixj > i) {
    bool f = (i & k) == 0;
    int t1 = table[i];
    int t2 = table[ixj];
    if (compare_rotations(f ? t2 : t1, f ? t1 : t2, genome, n)) {
      table[i] = t2;
      table[ixj] = t1;
    }
  }
}

__global__ void reconstruct_sequence(const int*__restrict table, const char*__restrict sequence, 
                                     char*__restrict transformed_sequence, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < n; i += stride) {
    transformed_sequence[i] = sequence[(n + table[i] - 1) % n];
  }
}

/* 
   returns a std::pair object
   the first item is the burrows wheeler transform of the input sequence in a std::string,
   the second item is the suffix array of the input sequence, represented as indicies of the given suffix, as an int*

   assumes input sequence already has ETX appended to it.
 */

std::pair<std::string,int*> bwt_with_suffix_array(const std::string sequence) {

  const int n = sequence.size();
  int table_size = sequence.size();
  // round the table size up to a power of 2 for bitonic sort
  table_size--;
  table_size |= table_size >> 1;
  table_size |= table_size >> 2;
  table_size |= table_size >> 4;
  table_size |= table_size >> 8;
  table_size |= table_size >> 16;
  table_size++;

  const int table_size_bytes = table_size * sizeof(int);
  const int seq_size_bytes = n * sizeof(char);

  int* d_table;
  cudaMalloc(&d_table, table_size_bytes);
  int* table = (int*) malloc(table_size_bytes);

  int numBlocks = (table_size + blockSize - 1) / blockSize;
  generate_table<<<numBlocks,blockSize>>>(d_table, table_size, n);

  char* d_sequence;
  cudaMalloc(&d_sequence, n * sizeof(char));
  cudaMemcpy(d_sequence, sequence.c_str(), n * sizeof(char), cudaMemcpyHostToDevice);
  for (int k = 2; k <= table_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      bitonic_sort_step<<<numBlocks,blockSize>>>(d_table, table_size, j, k, d_sequence, n);
    }
  }

  char* d_transformed_sequence;
  cudaMalloc(&d_transformed_sequence, seq_size_bytes);
  numBlocks = (n + blockSize - 1) / blockSize;
  reconstruct_sequence<<<numBlocks,blockSize>>>(d_table, d_sequence, d_transformed_sequence, n);
  char* transformed_sequence_cstr = (char*) malloc(seq_size_bytes);

  cudaMemcpy(transformed_sequence_cstr, d_transformed_sequence, seq_size_bytes, cudaMemcpyDeviceToHost);

  std::string transformed_sequence(transformed_sequence_cstr, n);

  cudaMemcpy(table, d_table, table_size_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_table);
  cudaFree(d_sequence);
  free(transformed_sequence_cstr);

  return std::make_pair(transformed_sequence, table);
}

std::string bwt(const std::string sequence) {
  auto data = bwt_with_suffix_array(sequence);
  free(data.second);
  return data.first;
}
