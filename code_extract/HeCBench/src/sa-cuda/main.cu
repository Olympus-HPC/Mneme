//  Created by gangliao on 12/22/14.
//  Copyright (c) 2014 gangliao. All rights reserved.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include "kernels.h"

#define MAX_ALPHA 26

// number of threads in a block
#define BLOCK_SIZE 256

void suffixArray(thrust::device_vector<int>&, thrust::device_vector<int>&, int, int);

#ifdef DEBUG
void print_suffix(const char *cc, int i)
{
  printf("%d: ", i);
  for (unsigned j = i; j < strlen(cc); j++)
    printf("%c", cc[j]);
  printf("\n");
}
#endif

bool read_data(char *filename, char *buffer, int num) {
  FILE *fh;
  fh = fopen(filename, "r");
  if (fh == NULL) {
    printf("Failed to open file %s\n", filename);
    return true;
  }
  fread(buffer, 1, num, fh);
  buffer[num] = '\0';
  fclose(fh);
  return false;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <dataset> <dataset size> <repeat>\n", argv[0]);
    printf("size of dataset to evaluate (10 - 1000000)\n");
    return 1;
  }

  char* filename = argv[1];

#ifdef DEBUG
  const int n = 100;
#else
  const int n = atoi(argv[2]);
#endif

  const int repeat = atoi(argv[3]);

  //load data buffer with a genome file
  char *data = (char *) malloc ((n + 1)*sizeof(char));
  bool fail = read_data(filename, data, n);
  if (data == NULL || fail) {
    if (data) free(data);
    return 1;
  }

  thrust::host_vector<int> h_inp (n + 3);
  thrust::host_vector<int> h_SA (n + 3, 0);
  thrust::device_vector<int> d_inp;
  thrust::device_vector<int> d_SA;

  int i;
  for (i = 0; i < n; i++) h_inp[i] = (int)data[i];
  h_inp[i] = 0; h_inp[i + 1] = 0; h_inp[i + 2] = 0; //prepare for triples

  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    d_inp = h_inp;
    d_SA = h_SA;

    suffixArray(d_inp, d_SA, n, MAX_ALPHA);

    h_SA = d_SA;
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Average suffix array construct time (input size = %d): %f (s)\n", 
         n, time.count() / repeat);

  long sum = 0;
  for (i = 0; i < n/2; i++)
    sum += (h_SA[2*i] + h_SA[2*i+1]) / abs(h_SA[2*i] - h_SA[2*i+1]);
  if (n % 2) sum -= h_SA[n-1];
  printf("checksum = 0x%lx\n", sum);
  
#ifdef DEBUG
  //peek sorted suffixes from data set
  for(i = 0; i < 8; i++) {
    printf("No.%d Index.", i); 
    print_suffix(data, h_SA[i]);
  }

  printf("...\n...\n");

  for(i = n-8; i < n; i++) {
    printf("No.%d Index.", i); 
    print_suffix(data, h_SA[i]);
  }
#endif

  free(data);
  return 0;
}

void suffixArray(thrust::device_vector<int>& s,
                 thrust::device_vector<int>& SA,
                 int n, int K)
{
  int n0 = (n + 2) / 3,
      n2 = n / 3,
      n02 = n0 + n2;

  thrust::device_vector<int> d_s12 (n02 + 3, 0);
  int *pd_s12 = thrust::raw_pointer_cast(&d_s12[0]);

  thrust::device_vector<int> d_SA12 (n02 + 3, 0);
  int *pd_SA12 = thrust::raw_pointer_cast(&d_SA12[0]);

  thrust::device_vector<int> d_s0 (n0, 0);
  int *pd_s0 = thrust::raw_pointer_cast(&d_s0[0]);

  thrust::device_vector<int> d_SA0 (n0, 0);
  int *pd_SA0 = thrust::raw_pointer_cast(&d_SA0[0]);

  thrust::device_vector<int> d_scan (n02 + 3);
  int *pd_scan = thrust::raw_pointer_cast(&d_scan[0]);

  int *pd_s = thrust::raw_pointer_cast(&s[0]);
  int *pd_SA = thrust::raw_pointer_cast(&SA[0]);

  dim3 numThreads(BLOCK_SIZE);
  dim3 numBlocks((n02 - 1) / BLOCK_SIZE + 1);

  // S12 initialization:
  Init_d_s12 <<<numBlocks, numThreads>>> (pd_s12, n02);

  // radix sort - using SA12 to store keys
  for (int i = 2; i >= 0; i--) {
    keybits <<<numBlocks, numThreads>>> (pd_SA12, pd_s12, pd_s, n02, i);
    thrust::sort_by_key(d_SA12.begin(), d_SA12.begin() + n02, d_s12.begin());
  }

  d_SA12 = d_s12;

  // stably sort the mod 0 suffixes from SA12 by their first character
  // find lexicographic names of triples

  InitScan <<<numBlocks, numThreads>>>(pd_s, pd_SA12, pd_scan, n02);

  thrust::exclusive_scan(d_scan.begin(), d_scan.begin() + n02 + 1, d_scan.begin());

  Set_suffix_rank <<<numBlocks, numThreads>>> (pd_s12, pd_SA12, pd_scan, n02, n0);

  int max_rank = d_scan[n02];

  // if max_rank is less than the size of s12, we have a repeat. repeat dc3.
  // else generate the suffix array of s12 directly
  if (max_rank < n02) {
    suffixArray(d_s12, d_SA12, n02, max_rank);
    Store_unique_ranks <<<numBlocks, numThreads>>> (pd_s12, pd_SA12, n02);
  }
  else {
    Compute_SA_From_UniqueRank <<<numBlocks, numThreads>>> (pd_s12, pd_SA12, n02);
  }

  InitScan2 <<<numBlocks, numThreads>>> (pd_SA12, pd_scan, n0, n02);
  thrust::exclusive_scan(d_scan.begin(), d_scan.begin() + n02, d_scan.begin()); 
  Set_S0 <<<numBlocks, numThreads>>> (pd_s0, pd_SA12, pd_scan, n0, n02);

  dim3 numBlocks2((n0 - 1) / BLOCK_SIZE + 1);
  keybits <<<numBlocks2, numThreads>>> (pd_SA0, pd_s0, pd_s, n0, 0);
  thrust::sort_by_key(d_SA0.begin(), d_SA0.begin() + n0, d_s0.begin());
  d_SA0 = d_s0;

  // merge sorted SA0 suffixes and sorted SA12 suffixes
  dim3 numBlocks3((n0 + n02 - 1) / BLOCK_SIZE + 1);
  merge_suffixes <<<numBlocks3, numThreads>>> (pd_SA0, pd_SA12, pd_SA, pd_s, pd_s12, n0, n02, n);
}
