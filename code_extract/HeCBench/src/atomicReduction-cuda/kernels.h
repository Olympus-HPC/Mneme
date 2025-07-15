__global__ void atomic_reduction(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx;i<arrayLength;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  atomicAdd(out,sum);
}

__global__ void atomic_reduction_v2(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx*2;i<arrayLength;i+=blockDim.x*gridDim.x*2) {
    sum+=in[i] + in[i+1];
  }
  atomicAdd(out,sum);
}

__global__ void atomic_reduction_v4(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx*4;i<arrayLength;i+=blockDim.x*gridDim.x*4) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3];
  }
  atomicAdd(out,sum);
}

__global__ void atomic_reduction_v8(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx*8;i<arrayLength;i+=blockDim.x*gridDim.x*8) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7];
  }
  atomicAdd(out,sum);
}

__global__ void atomic_reduction_v16(int *in, int* out, int arrayLength) {
  int sum=0;
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  for(int i= idx*16;i<arrayLength;i+=blockDim.x*gridDim.x*16) {
    sum+=in[i] + in[i+1] + in[i+2] + in[i+3] +in[i+4] +in[i+5] +in[i+6] +in[i+7]
      +in[i+8] +in[i+9] +in[i+10] +in[i+11] +in[i+12] +in[i+13] +in[i+14] +in[i+15] ;
  }
  atomicAdd(out,sum);
}

