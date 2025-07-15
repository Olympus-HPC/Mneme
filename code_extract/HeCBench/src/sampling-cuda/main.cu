/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "kernels.cu"

struct Dataset {
  int nrows_exact;
  int nrows_sampled;
  int ncols;
  int nrows_background;
  int max_samples;
  uint64_t seed;
};

typedef float T;

int main( int argc, char** argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int i, j, k;

  // each row represents a set of parameters for a testcase
  const std::vector<Dataset> inputs = {
    {1000, 0, 2000, 10, 11, 1234ULL},
    {0, 1000, 2000, 10, 11, 1234ULL},
    {1000, 1000, 2000, 10, 11, 1234ULL}
  };

  for (auto params : inputs) {

    double time = 0.0;

    for (int r = 0; r < repeat; r++) {
      // background
      T *b = (T*) malloc (sizeof(T) * params.nrows_background * params.ncols);
      // observation
      T *o = (T*) malloc (sizeof(T) * params.ncols);
      // nsamples
      int *n = (int*) malloc (sizeof(int) * params.nrows_sampled/2);
      
      int nrows_X = params.nrows_exact + params.nrows_sampled;
      float *X = (float*) malloc (sizeof(float) * nrows_X * params.ncols);
      T *d = (T*) malloc(sizeof(T) * nrows_X * params.nrows_background * params.ncols);

      // Assign a sentinel value to the observation to check easily later
      T sent_value = nrows_X * params.nrows_background * params.ncols * 100;
      for (i = 0; i < params.ncols; i++) {
        o[i] = sent_value;
      }

      // Initialize background array with different odd value per row, makes
      // it easier to debug if something goes wrong.
      for (i = 0; i < params.nrows_background; i++) {
        for (j = 0; j < params.ncols; j++) {
          b[i * params.ncols + j] = (i * 2) + 1;
        }
      }

      // Initialize the exact part of X. We create 2 `1` values per row for the test
      for (i = 0; i <  nrows_X * params.ncols; i++) X[i] = (float)0.0;
      for (i = 0; i < params.nrows_exact; i++) {
        for (j = i; j < i + 2; j++) {
          X[i * params.ncols + j] = (float)1.0;
        }
      }

      // Initialize the number of samples per row, we initialize each even row to
      // max samples and each odd row to max_samples - 1
      for (i = 0; i < params.nrows_sampled / 2; i++) {
        n[i] = params.max_samples - i % 2;
      }

      T *d_b;
      cudaMalloc((void**)&d_b, sizeof(T) * params.nrows_background * params.ncols);
      cudaMemcpy(d_b, b, sizeof(T) * params.nrows_background * params.ncols, cudaMemcpyHostToDevice );

      T *d_o;
      cudaMalloc((void**)&d_o, sizeof(T) * params.ncols);
      cudaMemcpy(d_o, o, sizeof(T) * params.ncols, cudaMemcpyHostToDevice );

      int *d_n;
      cudaMalloc((void**)&d_n, sizeof(int) * params.nrows_sampled/2);
      cudaMemcpy(d_n, n, sizeof(int) * params.nrows_sampled/2, cudaMemcpyHostToDevice);

      float *d_X;
      cudaMalloc((void**)&d_X, sizeof(float) * nrows_X * params.ncols);
      cudaMemcpy(d_X, X, sizeof(float) * nrows_X * params.ncols, cudaMemcpyHostToDevice);

      T *d_d;
      cudaMalloc((void**)&d_d, sizeof(T) * nrows_X * params.nrows_background * params.ncols);

      kernel_dataset(d_X, nrows_X, params.ncols, d_b,
          params.nrows_background, d_d, d_o, d_n,
          params.nrows_sampled, params.max_samples, params.seed,
          time);

      cudaMemcpy(X, d_X, sizeof(float) * nrows_X * params.ncols, cudaMemcpyDeviceToHost);
      cudaMemcpy(d, d_d, sizeof(T) * nrows_X * params.nrows_background * params.ncols, cudaMemcpyDeviceToHost);

      // Check the generated part of X by sampling. The first nrows_exact
      // correspond to the exact part generated before, so we just test after that.
      bool test_sampled_X = true;
      j = 0;
      int counter;

      for (i = params.nrows_exact * params.ncols; i < nrows_X * params.ncols / 2;
          i += 2 * params.ncols) {
        // check that number of samples is the number indicated by nsamples.
        counter = 0;
        for (k = i; k < i+params.ncols; k++)
          if (X[k] == 1) counter++;
        test_sampled_X = (test_sampled_X && (counter == n[j]));

        // check that number of samples of the next line is the compliment,
        // i.e. ncols - nsamples[j]
        counter = 0;
        for (k = i+params.ncols; k < i+2*params.ncols; k++)
          if (X[k] == 1) counter++;
        test_sampled_X = (test_sampled_X && (counter == (params.ncols - n[j])));
        j++;
      }

      // Check for the exact part of the generated dataset.
      bool test_scatter_exact = true;
      for (i = 0; i < params.nrows_exact; i++) {
        for (j = i * params.nrows_background * params.ncols;
            j < (i + 1) * params.nrows_background * params.ncols;
            j += params.ncols) {
          counter = 0;
          for (k = j; k < j+params.ncols; k++)
            if (d[k] == sent_value) counter++; 

          // Check that indeed we have two observation entries ber row
          test_scatter_exact = test_scatter_exact && (counter == 2);
          if (!test_scatter_exact) {
            printf("test_scatter_exact counter failed with: %d", counter);
            printf(", expected value was 2.\n");
            break;
          }
        }
        if (!test_scatter_exact) {
          break;
        }
      }

      // Check for the sampled part of the generated dataset
      bool test_scatter_sampled = true;

      // compliment_ctr is a helper counter to help check nrows_dataset per entry in
      // nsamples without complicating indexing since sampled part starts at nrows_sampled
      int compliment_ctr = 0;
      for (i = params.nrows_exact;
          i < params.nrows_exact + params.nrows_sampled / 2; i++) {
        // First set of dataset observations must correspond to nsamples[i]
        for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
            j <
            (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
            j += params.ncols) {

          counter = 0;
          for (k = j; k < j+params.ncols; k++)
            if (d[k] == sent_value) counter++; 

          test_scatter_sampled = test_scatter_sampled && (counter == n[i - params.nrows_exact]);
          if (!test_scatter_sampled) {
            printf("test_scatter_sampled counter failed with: %d", counter);
            printf(", expected value was %d.\n",  n[i - params.nrows_exact]);
            break;
          }
        }

        // The next set of samples must correspond to the compliment: ncols - nsamples[i]
        compliment_ctr++;
        for (j = (i + compliment_ctr) * params.nrows_background * params.ncols;
            j <
            (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
            j += params.ncols) {
          // Check that number of observation entries corresponds to nsamples.
          counter = 0;
          for (k = j; k < j+params.ncols; k++)
            if (d[k] == sent_value) counter++; 
          test_scatter_sampled = test_scatter_sampled &&
            (counter == params.ncols - n[i - params.nrows_exact]);
          if (!test_scatter_sampled) {
            printf("test_scatter_sampled counter failed with: %d", counter);
            printf(", expected value was %d.\n", params.ncols - n[i - params.nrows_exact]);
            break;
          }
        }
      }

      free(o);
      free(b);
      free(X);
      free(n);
      free(d);
      cudaFree(d_o);
      cudaFree(d_b);
      cudaFree(d_X);
      cudaFree(d_n);
      cudaFree(d_d);
    }

    printf("Average execution time of kernels: %f (us)\n", (time * 1e-3) / repeat);
  }

  return 0;
}
