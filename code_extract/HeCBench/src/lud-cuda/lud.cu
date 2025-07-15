#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <chrono>
#include <cuda.h>
#include "common.h"

#define BLOCK_SIZE 16

#include "lud_kernels.cu"

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

int main ( int argc, char *argv[] )
{
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'i':
        input_file = optarg;
        break;
      case 'v':
        do_verify = 1;
        break;
      case 's':
        matrix_dim = atoi(optarg);
        printf("Generate input matrix internally, size=%d\n", matrix_dim);
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }  

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } 

  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }

  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  /* beginning of timing point */
  stopwatch_start(&sw);

  float *d_m;
  cudaMalloc((void**)&d_m, matrix_dim*matrix_dim*sizeof(float));
  cudaMemcpy(d_m, m, matrix_dim*matrix_dim*sizeof(float), cudaMemcpyHostToDevice);

  int offset;
  int i=0;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
    offset = i;  // add the offset 
    lud_diagonal<<<1, BLOCK_SIZE>>>(d_m, matrix_dim, offset);
    lud_perimeter<<<(matrix_dim-i)/BLOCK_SIZE-1, 2*BLOCK_SIZE>>>(d_m, matrix_dim, offset);
    lud_internal<<< dim3((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1),
	    dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_m, matrix_dim, offset);
  } // for

  offset = i;  // add the offset 
  lud_diagonal<<<1, BLOCK_SIZE>>>(d_m, matrix_dim, offset);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time : %f (s)\n", time * 1e-9f);

  cudaMemcpy(m, d_m, matrix_dim*matrix_dim*sizeof(float), cudaMemcpyDeviceToHost);

  /* end of timing point */
  stopwatch_stop(&sw);
  printf("Device offloading time (s): %lf\n", get_interval_by_sec(&sw));

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);
  cudaFree(d_m);
  return 0;
}
