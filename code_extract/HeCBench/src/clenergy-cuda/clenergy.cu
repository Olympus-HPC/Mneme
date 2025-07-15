/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
 * GPU accelerated coulombic potential grid test code
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "WKFUtils.h"

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}

#define MAXATOMS 4000

#define UNROLLX       8
#define UNROLLY       1
#define BLOCKSIZEX    8
#define BLOCKSIZEY    8 
#define BLOCKSIZE    BLOCKSIZEX * BLOCKSIZEY

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
__global__ void cenergy(const int numatoms, const float gridspacing, 
                        float *energygrid, const float4 *atominfo) 
{
  unsigned int xindex  = blockIdx.x * blockDim.x * UNROLLX + threadIdx.x;
  unsigned int yindex  = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int outaddr = gridDim.x * blockDim.x * UNROLLX * yindex + xindex;

  float coory = gridspacing * yindex;
  float coorx = gridspacing * xindex;

  float energyvalx1=0.0f;
  float energyvalx2=0.0f;
  float energyvalx3=0.0f;
  float energyvalx4=0.0f;
  float energyvalx5=0.0f;
  float energyvalx6=0.0f;
  float energyvalx7=0.0f;
  float energyvalx8=0.0f;

  float gridspacing_u = gridspacing * BLOCKSIZEX;

  //
  // XXX 59/8 FLOPS per atom
  //
  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dyz2 = (dy * dy) + atominfo[atomid].z;

    float dx1 = coorx - atominfo[atomid].x;
    float dx2 = dx1 + gridspacing_u;
    float dx3 = dx2 + gridspacing_u;
    float dx4 = dx3 + gridspacing_u;
    float dx5 = dx4 + gridspacing_u;
    float dx6 = dx5 + gridspacing_u;
    float dx7 = dx6 + gridspacing_u;
    float dx8 = dx7 + gridspacing_u;

    energyvalx1 += atominfo[atomid].w * rsqrtf(dx1*dx1 + dyz2);
    energyvalx2 += atominfo[atomid].w * rsqrtf(dx2*dx2 + dyz2);
    energyvalx3 += atominfo[atomid].w * rsqrtf(dx3*dx3 + dyz2);
    energyvalx4 += atominfo[atomid].w * rsqrtf(dx4*dx4 + dyz2);
    energyvalx5 += atominfo[atomid].w * rsqrtf(dx5*dx5 + dyz2);
    energyvalx6 += atominfo[atomid].w * rsqrtf(dx6*dx6 + dyz2);
    energyvalx7 += atominfo[atomid].w * rsqrtf(dx7*dx7 + dyz2);
    energyvalx8 += atominfo[atomid].w * rsqrtf(dx8*dx8 + dyz2);
  }

  energygrid[outaddr             ] += energyvalx1;
  energygrid[outaddr+1*BLOCKSIZEX] += energyvalx2;
  energygrid[outaddr+2*BLOCKSIZEX] += energyvalx3;
  energygrid[outaddr+3*BLOCKSIZEX] += energyvalx4;
  energygrid[outaddr+4*BLOCKSIZEX] += energyvalx5;
  energygrid[outaddr+5*BLOCKSIZEX] += energyvalx6;
  energygrid[outaddr+6*BLOCKSIZEX] += energyvalx7;
  energygrid[outaddr+7*BLOCKSIZEX] += energyvalx8;
}



int copyatoms(float *atoms, int count, float zplane, float4 *atominfo) {
  CUERR // check and clear any existing errors

  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  float atompre[4*MAXATOMS];
  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  cudaMemcpy((float*)atominfo, atompre, count * 4 * sizeof(float), cudaMemcpyHostToDevice);
  CUERR // check and clear any existing errors

  return 0;
}


int initatoms(float **atombuf, int count, dim3 volsize, float gridspacing) {
  float3 size;
  int i;
  float *atoms;
  srand(2);

  atoms = (float *) malloc(count * 4 * sizeof(float));
  *atombuf = atoms;

  // compute grid dimensions in angstroms
  size.x = gridspacing * volsize.x;
  size.y = gridspacing * volsize.y;
  size.z = gridspacing * volsize.z;

  for (i=0; i<count; i++) {
    int addr = i * 4;
    atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
    atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
    atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
    atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
  }  

  return 0;
}


int main(int argc, char** argv) {
  float *doutput = NULL;
  float4 *datominfo = NULL;
  float *energy = NULL;
  float *atoms = NULL;
  dim3 volsize, Gsz, Bsz;
  wkf_timerhandle runtimer, mastertimer, copytimer, hostcopytimer;
  float copytotal, runtotal, mastertotal, hostcopytotal;
  const char *statestr = "|/-\\.";
  int state=0;

  printf("GPU accelerated coulombic potential microbenchmark\n");
  printf("--------------------------------------------------------\n");
  printf("  Single-threaded single-device test run.\n");

  // number of atoms to simulate
  int atomcount = 1000000;

  // setup energy grid size
  // XXX this is a large test case to clearly illustrate that even while
  //     the CUDA kernel is running entirely on the GPU, the CUDA runtime
  //     library is soaking up the entire host CPU for some reason.
  volsize.x = 768;
  volsize.y = 768;
  volsize.z = 1;

  // set voxel spacing
  float gridspacing = 0.1;

  // setup CUDA grid and block sizes
  // XXX we have to make a trade-off between the number of threads per
  //     block and the resulting padding size we'll end up with since
  //     each thread will do several consecutive grid cells in this version,
  //     we're using up some of our available parallelism to reduce overhead.
  Bsz.x = BLOCKSIZEX;
  Bsz.y = BLOCKSIZEY;
  Bsz.z = 1;
  Gsz.x = volsize.x / (Bsz.x * UNROLLX);
  Gsz.y = volsize.y / (Bsz.y * UNROLLY); 
  Gsz.z = volsize.z; 

  // initialize the wall clock timers
  runtimer = wkf_timer_create();
  mastertimer = wkf_timer_create();
  copytimer = wkf_timer_create();
  hostcopytimer = wkf_timer_create();
  copytotal = 0;
  runtotal = 0;
  hostcopytotal = 0;

  printf("Grid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);
  printf("Running kernel(atoms:%d, gridspacing %g, z %d)\n", atomcount, gridspacing, 0);

  // allocate and initialize atom coordinates and charges
  if (initatoms(&atoms, atomcount, volsize, gridspacing))
    return -1;

  // allocate and initialize the GPU output array
  int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;
  printf("Allocating %.2fMB of memory for output buffer...\n", volmemsz / (1024.0 * 1024.0));

  cudaMalloc((void**)&doutput, volmemsz);
  cudaMalloc((void**)&datominfo, sizeof(float4) * MAXATOMS);
  CUERR // check and clear any existing errors
  cudaMemset(doutput, 0, volmemsz);
  CUERR // check and clear any existing errors

  printf("starting run...\n");
  wkf_timer_start(mastertimer);

  int iterations=0;
  int atomstart;
  for (atomstart=0; atomstart<atomcount; atomstart+=MAXATOMS) {   
    iterations++;
    int runatoms;
    int atomsremaining = atomcount - atomstart;
    if (atomsremaining > MAXATOMS)
      runatoms = MAXATOMS;
    else
      runatoms = atomsremaining;

    printf("%c\r", statestr[state]);
    fflush(stdout);
    state = (state+1) & 3;

    // copy the atoms to the GPU
    wkf_timer_start(copytimer);
    if (copyatoms(atoms + 4*atomstart, runatoms, 0*gridspacing, datominfo)) 
      return -1;
    wkf_timer_stop(copytimer);
    copytotal += wkf_timer_time(copytimer);
 
    // RUN the kernel...
    wkf_timer_start(runtimer);
    cenergy<<<Gsz, Bsz, 0>>>(runatoms, 0.1, doutput, datominfo);
    cudaDeviceSynchronize();
    CUERR // check and clear any existing errors
    wkf_timer_stop(runtimer);
    runtotal += wkf_timer_time(runtimer);
  }
  printf("Done\n");

  wkf_timer_stop(mastertimer);
  mastertotal = wkf_timer_time(mastertimer);

  // Copy the GPU output data back to the host and use/store it..
  energy = (float *) malloc(volmemsz);
  wkf_timer_start(hostcopytimer);
  cudaMemcpy(energy, doutput, volmemsz,  cudaMemcpyDeviceToHost);
  CUERR // check and clear any existing errors
  wkf_timer_stop(hostcopytimer);
  hostcopytotal=wkf_timer_time(hostcopytimer);

  int i, j;
  for (j=0; j<8; j++) {
    for (i=0; i<8; i++) {
      int addr = j*volsize.x + i;
      printf("[%d] %.1f ", addr, energy[addr]);
    }
    printf("\n");
  }

  printf("Final calculation required %d iterations of %d atoms\n", iterations, MAXATOMS);
  printf("Copy time: %f seconds, %f per iteration\n", copytotal, copytotal / (float) iterations);
  printf("Kernel time: %f seconds, %f per iteration\n", runtotal, runtotal / (float) iterations);
  printf("Total time: %f seconds\n", mastertotal);
  printf("Kernel invocation rate: %f iterations per second\n", iterations / mastertotal);
  printf("GPU to host copy bandwidth: %gMB/sec, %f seconds total\n",
         (volmemsz / (1024.0 * 1024.0)) / hostcopytotal, hostcopytotal);

  double atomevalssec = ((double) volsize.x * volsize.y * volsize.z * atomcount) / (mastertotal * 1000000000.0);
  printf("Efficiency metric, %g billion atom evals per second\n", atomevalssec);

  /* 59/8 FLOPS per atom eval */
  printf("FP performance: %g GFLOPS\n", atomevalssec * (59.0/8.0));
  
  free(atoms);
  free(energy);
  cudaFree(doutput);
  cudaFree(datominfo);

  return 0;
}
