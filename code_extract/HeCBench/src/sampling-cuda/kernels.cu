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

/*
* Kernel distributes exact part of the kernel shap dataset
* Each block scatters the data of a row of `observations` into the (number of rows of
* background) in `dataset`, based on the row of `X`.
* So, given:
* background = [[0, 1, 2],
                [3, 4, 5]]
* observation = [100, 101, 102]
* X = [[1, 0, 1],
*      [0, 1, 1]]
*
* dataset (output):
* [[100, 1, 102],
*  [100, 4, 102]
*  [0, 101, 102],
*  [3, 101, 102]]
*
*
*/

template <typename DataT, typename IdxT>
__global__
void exact_rows_kernel(
  float*__restrict__ X,
  const IdxT nrows_X,
  const IdxT ncols,
  const DataT*__restrict__ background,
  const IdxT nrows_background,
  DataT*__restrict__ dataset,
  const DataT*__restrict__ observation)
{
  // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure data coelescing
  int col = threadIdx.x;
  int row = blockIdx.x * ncols;

  while (col < ncols) {
    // Load the X idx for the current column
    int curr_X = (int)X[row + col];

    // Iterate over nrows_background
    for (int row_idx = blockIdx.x * nrows_background;
         row_idx < blockIdx.x * nrows_background + nrows_background;
         row_idx += 1) {
      if (curr_X == 0) {
        dataset[row_idx * ncols + col] =
          background[(row_idx % nrows_background) * ncols + col];
      } else {
        dataset[row_idx * ncols + col] = observation[col];
      }
    }
    // Increment the column
    col += blockDim.x;
  }
}

/*
* Kernel distributes sampled part of the kernel shap dataset
* The first thread of each block calculates the sampling of `k` entries of `observation`
* to scatter into `dataset`. Afterwards each block scatters the data of a row of `X` into the (number of rows of
* background) in `dataset`.
* So, given:
* background = [[0, 1, 2, 3],
                [5, 6, 7, 8]]
* observation = [100, 101, 102, 103]
* nsamples = [3, 2]
*
* X (output)
*      [[1, 0, 1, 1],
*       [0, 1, 1, 0]]
*
* dataset (output):
* [[100, 1, 102, 103],
*  [100, 6, 102, 103]
*  [0, 101, 102, 3],
*  [5, 101, 102, 8]]
*
*
*/
__device__
double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}  

template <typename DataT, typename IdxT>
__global__
void sampled_rows_kernel(
  const IdxT*__restrict__ nsamples,
  float*__restrict__ X,
  const IdxT nrows_X,
  const IdxT ncols,
  DataT*__restrict__ background,
  const IdxT nrows_background,
  DataT*__restrict__ dataset,
  const DataT*__restrict__ observation,
  uint64_t seed)
{
  // int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // see what k this block will generate
  int k_blk = nsamples[blockIdx.x];

  // First k threads of block generate samples
  if (threadIdx.x < k_blk) {
    int rand_idx = (int)(LCG_random_double(&seed) * ncols);

    // Since X is initialized to 0, we quickly check for collisions (if k_blk << ncols the likelyhood of collisions is low)
    while (atomicExch(&(X[2 * blockIdx.x * ncols + rand_idx]), 1) == 1) {
      rand_idx = (int)(LCG_random_double(&seed) * ncols);
    }
  }
  __syncthreads();

  // Each block processes one row of X. Columns are iterated over by blockDim.x at a time to ensure data coelescing
  int col_idx = threadIdx.x;
  while (col_idx < ncols) {
    // Load the X idx for the current column
    int curr_X = (int)X[2 * blockIdx.x * ncols + col_idx];
    X[(2 * blockIdx.x + 1) * ncols + col_idx] = 1 - curr_X;

    for (int bg_row_idx = 2 * blockIdx.x * nrows_background;
         bg_row_idx < 2 * blockIdx.x * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx % nrows_background) * ncols + col_idx];
      } else {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      }
    }

    for (int bg_row_idx = (2 * blockIdx.x + 1) * nrows_background;
         bg_row_idx <
         (2 * blockIdx.x + 1) * nrows_background + nrows_background;
         bg_row_idx += 1) {
      if (curr_X == 0) {
        dataset[bg_row_idx * ncols + col_idx] = observation[col_idx];
      } else {
        // if(threadIdx.x == 0) printf("tid bg_row_idx: %d %d\n", tid, bg_row_idx);
        dataset[bg_row_idx * ncols + col_idx] =
          background[(bg_row_idx) % nrows_background * ncols + col_idx];
      }
    }

    col_idx += blockDim.x;
  }
}

template <typename DataT, typename IdxT>
void kernel_dataset(float* X,
                    const IdxT nrows_X,
                    const IdxT ncols,
                    DataT* background,
                    const IdxT nrows_background,
                    DataT* dataset,
                    DataT* observation,
                    int* nsamples,
                    const int len_samples,
                    const int maxsample,
                    const uint64_t seed,
                    double &time) 
{
  IdxT nblks;
  IdxT nthreads;

  nthreads = std::min(256, ncols);
  nblks = nrows_X - len_samples;
  //printf("nblks = %d len_samples = %d\n", nblks, len_samples );

  auto start = std::chrono::steady_clock::now();

  if (nblks > 0) {
    exact_rows_kernel<<<nblks, nthreads>>>(
      X, nrows_X, ncols, background, nrows_background, dataset, observation);
  }

  // check if random part of the dataset is needed
  if (len_samples > 0) {
    nblks = len_samples / 2;
    // each block does a sample and its compliment
    sampled_rows_kernel<<<nblks, nthreads>>>(
      nsamples, &X[(nrows_X - len_samples) * ncols], len_samples, ncols,
      background, nrows_background,
      &dataset[(nrows_X - len_samples) * nrows_background * ncols], observation,
      seed);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
