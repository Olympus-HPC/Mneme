#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include <nccl.h>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
  do{                                                                                     \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
  }while(0)


__global__
void test(double *d, const long int n) {
  for (long i = blockDim.x * blockIdx.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    d[i] = d[i] + 1;
  }
}


int main(int argc, char *argv[])
{
  /* -------------------------------------------------------------------------------------------
     MPI Initialization 
     --------------------------------------------------------------------------------------------*/
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(size != 2){
    if(rank == 0){
      printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
    }
    MPI_Finalize();
    exit(0);
  }

  // Map MPI ranks to GPUs
  int num_devices = 0;
  cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
  cudaErrorCheck( cudaSetDevice(rank % num_devices) );

  //initialize NCCL
  ncclComm_t comm;
  ncclUniqueId id;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
  cudaStream_t stream;
  cudaErrorCheck(cudaStreamCreate(&stream));

  //   Loop from 65536 B to 1 GB
  for(int i=16; i<=27; i++){

    long int N = 1 << i;

    double *h_A, *d_A;
    h_A = (double*) malloc (N*sizeof(double)); 
    cudaErrorCheck( cudaMalloc((void**)&d_A, N*sizeof(double)) );
    cudaErrorCheck( cudaMemset(d_A, 0, N*sizeof(double)) );
    cudaErrorCheck( cudaDeviceSynchronize() );

    int loop_count = 50;

    // Warm-up and validate NCCL pingpong
    for(int i=1; i<=5; i++){
      if(rank == 0){
        NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 1, comm, stream));
        NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 1, comm, stream));
      }
      else if(rank == 1){
        NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 0, comm, stream));
        test<<<1024, 256, 0, stream>>>(d_A, N);
        cudaErrorCheck( cudaStreamSynchronize(stream) );
        NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 0, comm, stream));
      }
    }
    cudaErrorCheck(cudaStreamSynchronize(stream));
    if(rank == 0) {
      cudaErrorCheck(cudaMemcpy(h_A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost));
      for (long int i = 0; i < N; i++) {
        if(h_A[i] != 5) {
          printf("ERROR: NCCL pingpong test failed: %lf\n", h_A[i]);
          break;
        }
      }
    }

    free(h_A);

    // Time loop_count iterations of data transfer size 8*N bytes
    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    for(int i=1; i<=loop_count; i++){
      if(rank == 0){
        NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 1, comm, stream));
        NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 1, comm, stream));
      }
      else if(rank == 1){
        NCCLCHECK(ncclRecv(d_A, N, ncclFloat64, 0, comm, stream));
        NCCLCHECK(ncclSend(d_A, N, ncclFloat64, 0, comm, stream));
      }
    }
    cudaErrorCheck(cudaStreamSynchronize(stream));

    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    long int num_B = 8*N;
    double num_GB = (double)num_B / 1.0e9;
    double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

    if(rank == 0)
      printf("NCCL: Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

    cudaErrorCheck( cudaFree(d_A) );
  }

  cudaErrorCheck(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  MPI_Finalize();

  return 0;
}
