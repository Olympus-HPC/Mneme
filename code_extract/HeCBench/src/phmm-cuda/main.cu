#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include <cuda.h>
#include "constants_types.h"
#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  fArray *d_cur_forward;
  fArray *d_next_forward;
  fArray *d_emis;
  tArray *d_trans;
  lArray *d_like;
  sArray *d_start;

  dim3 dimGrid(batch);
  dim3 dimBlock(states-1);

  size_t forward_matrix_size = (x_dim+1)*(y_dim+1)*batch*(states-1)*sizeof(double);
  size_t emissions_size = (x_dim+1)*(y_dim+1)*batch*(states-1)*sizeof(double);
  size_t transitions_size = (x_dim+1)*(states-1)*states*batch*sizeof(double);
  size_t start_transitions_size = batch*(states-1)*sizeof(double);
  size_t likelihood_size = 2*2*(states-1)*batch*sizeof(double);

  fArray *h_cur_forward = (fArray*) malloc (forward_matrix_size); 
  fArray *h_emis = (fArray*) malloc (emissions_size);
  tArray *h_trans = (tArray*) malloc (transitions_size);
  lArray *h_like = (lArray*) malloc (likelihood_size);
  sArray *h_start = (sArray*) malloc (start_transitions_size);

  std::default_random_engine rng (123);
  std::uniform_real_distribution<double> dist (0.0, 1.0);
  for (int i = 0; i < x_dim+1; i++) {
    for (int j = 0; j < y_dim+1; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states-1; s++) {
           h_cur_forward[i][j][b][s] = dist(rng);
           h_emis[i][j][b][s] = dist(rng);
        }
      }
    }
  }

  for (int i = 0; i < x_dim+1; i++) {
    for (int b = 0; b < batch; b++) {
      for (int s = 0; s < states-1; s++) {
        for (int t = 0; t < states; t++) {
          h_trans[i][b][s][t] = dist(rng);
        }
      }
    }
  }
         
  for (int i = 0; i < batch; i++) {
    for (int s = 0; s < states-1; s++) {
      h_start[i][s] = dist(rng);
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j< 2; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states-1; s++) {
          h_like[i][j][b][s] = dist(rng);
        }
      }
    }
  }

  cudaMalloc((void**)&d_cur_forward, forward_matrix_size); 
  cudaMemcpy(d_cur_forward, h_cur_forward, forward_matrix_size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_next_forward, forward_matrix_size);  

  cudaMalloc((void**)&d_emis, emissions_size);
  cudaMemcpy(d_emis, h_emis, forward_matrix_size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_trans, transitions_size);
  cudaMemcpy(d_trans, h_trans, transitions_size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_like, likelihood_size);
  cudaMemcpy(d_like, h_like, likelihood_size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_start, start_transitions_size);
  cudaMemcpy(d_start, h_start, start_transitions_size, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  auto t1 = std::chrono::high_resolution_clock::now();

  for(int count = 0; count < repeat; count++) {
    for (int i = 1; i < x_dim + 1; i++) {
      for (int j = 1; j < y_dim + 1; j++) {
        pair_HMM_forward<<<dimGrid, dimBlock>>>(i, j, d_cur_forward, d_trans, d_emis, d_like, d_start, d_next_forward);
        auto t = d_cur_forward;
        d_cur_forward = d_next_forward;
        d_next_forward = t;
      }
    }
  }
  cudaDeviceSynchronize();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> milli = (t2 - t1);
  std::cout << "Total execution time " <<  milli.count() << " milliseconds\n" ;

  cudaMemcpy(h_cur_forward, d_cur_forward, forward_matrix_size, cudaMemcpyDeviceToHost);

  double checkSum = 0.0;
  for (int i = 0; i < x_dim+1; i++) {
    for (int j = 0; j < y_dim+1; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states-1; s++) {
          #ifdef DEBUG
          std::cout << h_cur_forward[i][j][b][s] << std::endl;
          #endif
          checkSum += h_cur_forward[i][j][b][s];
        }
      }
    }
  }
  std::cout << "Checksum " << checkSum << std::endl;

  cudaFree(d_cur_forward);
  cudaFree(d_next_forward);
  cudaFree(d_emis);
  cudaFree(d_trans);
  cudaFree(d_like);
  cudaFree(d_start);
  free(h_cur_forward);
  free(h_emis);
  free(h_trans);
  free(h_like);
  free(h_start);

  return 0;
}
