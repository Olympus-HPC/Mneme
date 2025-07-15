#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclComm_t comm;
  ncclUniqueId id;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  //picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));

  //initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  float *sendbuff, *recvbuff;
  float *h_sendbuff, *h_recvbuff;
  double start_time, stop_time, elapsed_time;

  for (int size = 1024*1024; size <= 1000 * 1024 * 1024; size = size * 10) {

    h_sendbuff = (float*) malloc (size * sizeof(float));
    for (int i = 0; i < size; i++) h_sendbuff[i] = 1;

    h_recvbuff = (float*) malloc (size * sizeof(float));

    CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff, h_sendbuff, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));

    cudaStream_t s;
    CUDACHECK(cudaStreamCreate(&s));

    start_time = MPI_Wtime();

    //communicating using NCCL
    for (int i = 0; i < repeat; i++) {
      NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat, ncclSum,
          comm, s));
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    CUDACHECK(cudaStreamSynchronize(s));

    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    if (myRank == 0) {
      long int num_B = sizeof(float) * size * nRanks;
      long int B_in_GB = 1 << 30;
      double num_GB = (double)num_B / (double)B_in_GB;
      double avg_time_per_transfer = elapsed_time / repeat;

      printf("Transfer size (B): %10li, Average Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );
    }

    CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff, size * sizeof(float), cudaMemcpyDeviceToHost));

    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (h_recvbuff[i] != float(nRanks)) {
         ok = false;
         break;
      }
    }
    free(h_sendbuff);
    free(h_recvbuff);

    printf("MPI Rank %d: %s\n", myRank, ok ? "PASS" : "FAIL");
  }

  //finalizing NCCL
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  return 0;
}
