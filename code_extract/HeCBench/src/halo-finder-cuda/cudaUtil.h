#include <cuda.h>
#pragma once
#include <stdio.h>

#define cudaCheckError() {                                                       \
 cudaError_t e=cudaGetLastError();                                                \
 if(e!=cudaSuccess) {                                                            \
   printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));  \
   exit(0);                                                                     \
 }                                                                              \
}

__device__  __forceinline__
void atomicWarpReduceAndUpdate(POSVEL_T *out, POSVEL_T val) {
  //perform shfl reduction
  val+=__shfl_down(val, 16); 
  val+=__shfl_down(val, 8); 
  val+=__shfl_down(val, 4);
  val+=__shfl_down(val, 2); 
  val+=__shfl_down(val, 1);

  if(threadIdx.x%32==0)
    atomicAdd(out,val);  //atomics are unecessary but they are faster than non-atomics due to a single bus transaction
}

class cudaDeviceSelector {
  public:
  cudaDeviceSelector() {
    char* str;
    int local_rank = 0;
    int numDev=1;

    //No MPI at this time so go by enviornment variables. 
    //This may need to be updated to match your MPI flavor
    if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
    }
    else if((str = getenv("SLURM_LOCALID")) != NULL) {
      local_rank = atoi(str);
    }

    //get the number of devices to use
    if((str = getenv("HACC_NUM_CUDA_DEV")) != NULL) {
      numDev=atoi(str);
    }

#if 0

#if 0
    //Use MPS,  need to figure out how to set numDev, perhaps and enviornment varaible?
    char var[100];
    sprintf(var,"/tmp/nvidia-mps_%d",local_rank%numDev);
    setenv("CUDA_MPS_PIPE_DIRECTORY",var,1);
#endif
#else 
    int dev;
    //set via local MPI rank 
    dev = local_rank % numDev;
 
    //we must set this for all threads
    cudaSetDevice(dev);
#endif
  }
};
