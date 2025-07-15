#include <chrono>
#include <cuda.h>
#include "kernel.h"

__device__ float fitness_function(float x[])
{
  float y1 = F(x[0]);
  float yn = F(x[DIM-1]);
  float res = powf(sinf(phi*y1), 2.f) + powf(yn-1, 2.f);

  for(int i = 0; i < DIM-1; i++)
  {
    float y = F(x[i]);
    float yp = F(x[i+1]);
    res += powf(y-1.f, 2.f) * (1.f + 10.f * powf(sinf(phi*yp), 2.f));
  }

  return res;
}

__global__
void kernelUpdateParticle(float *__restrict__ positions,
                          float *__restrict__ velocities,
                          const float *__restrict__ pBests,
                          const float *__restrict__ gBest,
                          const int p,
                          const float rp,
                          const float rg)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= p*DIM) return;

  velocities[i]=OMEGA*velocities[i]+
                c1*rp*(pBests[i]-positions[i])+
                c2*rg*(gBest[i%DIM]-positions[i]);
  positions[i]+=velocities[i];
}

__global__
void kernelUpdatePBest(const float *__restrict__ positions,
                             float *__restrict__ pBests,
                             float *__restrict__ gBest,
                       const int p)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= p) return;
  i = i*DIM;

  float tempParticle1[DIM];
  float tempParticle2[DIM];

  for(int j=0;j<DIM;j++)
  {
    tempParticle1[j]=positions[i+j];
    tempParticle2[j]=pBests[i+j];
  }

  if(fitness_function(tempParticle1)<fitness_function(tempParticle2))
  {
    for(int j=0;j<DIM;j++)
      pBests[i+j]=tempParticle1[j];

    if(fitness_function(tempParticle1)<130.f) //fitness_function(gBest))
    {
      for(int j=0;j<DIM;j++) {
        atomicAdd(gBest+j,tempParticle1[j]);
      }
    }
  }
}

extern "C" void gpu_pso(int p, int r,
                        float *positions,float *velocities,float *pBests,float *gBest)
{
  int size = p*DIM;
  size_t size_byte = sizeof(float) * size;
  size_t res_size_byte = sizeof(float) * DIM;

  float *devPos;
  float *devVel;
  float *devPBest;
  float *devGBest;

  cudaMalloc((void**)&devPos,size_byte);
  cudaMalloc((void**)&devVel,size_byte);
  cudaMalloc((void**)&devPBest,size_byte);
  cudaMalloc((void**)&devGBest,res_size_byte);

  int threadNum=256;
  int blocksNum1=(size+threadNum-1)/threadNum;
  int blocksNum2=(p+threadNum-1)/threadNum;

  cudaMemcpy(devPos,positions,size_byte,cudaMemcpyHostToDevice);
  cudaMemcpy(devVel,velocities,size_byte,cudaMemcpyHostToDevice);
  cudaMemcpy(devPBest,pBests,size_byte,cudaMemcpyHostToDevice);
  cudaMemcpy(devGBest,gBest,res_size_byte,cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for(int iter=0;iter<r;iter++)
  {
    float rp=getRandomClamped(iter);
    float rg=getRandomClamped(r-iter);
    kernelUpdateParticle<<<blocksNum1,threadNum>>>(
      devPos,devVel,devPBest,devGBest,
      p,rp,rg);

    kernelUpdatePBest<<<blocksNum2,threadNum>>>(devPos,devPBest,devGBest,p);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", time * 1e-3f / r);
  
  cudaMemcpy(gBest,devGBest,res_size_byte,cudaMemcpyDeviceToHost);
  cudaMemcpy(pBests,devPBest,size_byte,cudaMemcpyDeviceToHost);

  cudaFree(devPos);
  cudaFree(devVel);
  cudaFree(devPBest);
  cudaFree(devGBest);
}
