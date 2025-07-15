/*
   Illinois Open Source License

   University of Illinois/NCSA
   Open Source License

   Copyright © 2009,    University of Illinois.  All rights reserved.

   Developed by: 
   Innovative Systems Lab
   National Center for Supercomputing Applications
http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.

 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.

 * Neither the names of Innovative Systems Lab and National Center for Supercomputing Applications, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */


// Angular correlation function kernel
// Takes two sets of cartesians in g_idata1, g_idata2,
// computes dot products for all pairs, uses waterfall search
// to determine the appropriate bin for each dot product,
// and outputs bins in g_odata (packed 4 bins to 1 unsigned int)

// The problem is treated as a grid of dot products.
// Each thread block has 128 threads, and calculates the dot
// products for a 128x128 sub-grid.

#ifndef _ACF_KERNEL_H_
#define _ACF_KERNEL_H_

#define LOG2_GRID_SIZE 14

__device__ __constant__ double binbounds[NUMBINS-1];

// Similar to ACF kernel, but takes advantage of symmetry to cut computations down by half.
// Obviously, due to symmetry, it needs only one input set.
__global__ void ACFKernelSymm(cartesian g_idata1, unsigned int* g_odata)
{
  extern __shared__ double3 sdata[];
  int tx = (blockIdx.x<<7) + threadIdx.x;
  int by = (blockIdx.y<<7);
  if(blockIdx.x < blockIdx.y) {    // All elements computed by block are above the main diagonal
    by <<= (LOG2_GRID_SIZE - 2);
    by += tx;
#pragma unroll
    for(int i=0; i<128; i+=4) {
      g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = 2088533116; //  (124<<24) + (124<<16) + (124<<8) + (124);
    }
  }
  else if(blockIdx.x > blockIdx.y) {  // All elements computed by block are below the main diagonal
    double temp;
    unsigned int temp2;
    double3 vec1, vec2;

    vec1.x = g_idata1.x[tx];
    vec1.y = g_idata1.y[tx];
    vec1.z = g_idata1.z[tx];
    sdata[threadIdx.x].x = g_idata1.x[by+threadIdx.x];
    sdata[threadIdx.x].y = g_idata1.y[by+threadIdx.x];
    sdata[threadIdx.x].z = g_idata1.z[by+threadIdx.x];

    __syncthreads();

    by <<= (LOG2_GRID_SIZE - 2);
    by += tx;

#pragma unroll
    for(int i=0; i<128; i+=4) {
      temp2 = 0;
#pragma unroll
      for(int j=0; j<4; j++) {
        vec2 = sdata[i+j];
        temp = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
        if(temp < binbounds[30]) temp2 += (124<<(j<<3));
        else if(temp < binbounds[29]) temp2 += (120<<(j<<3));
        else if(temp < binbounds[28]) temp2 += (116<<(j<<3));
        else if(temp < binbounds[27]) temp2 += (112<<(j<<3));
        else if(temp < binbounds[26]) temp2 += (108<<(j<<3));
        else if(temp < binbounds[25]) temp2 += (104<<(j<<3));
        else if(temp < binbounds[24]) temp2 += (100<<(j<<3));
        else if(temp < binbounds[23]) temp2 += (96<<(j<<3));
        else if(temp < binbounds[22]) temp2 += (92<<(j<<3));
        else if(temp < binbounds[21]) temp2 += (88<<(j<<3));
        else if(temp < binbounds[20]) temp2 += (84<<(j<<3));
        else if(temp < binbounds[19]) temp2 += (80<<(j<<3));
        else if(temp < binbounds[18]) temp2 += (76<<(j<<3));
        else if(temp < binbounds[17]) temp2 += (72<<(j<<3));
        else if(temp < binbounds[16]) temp2 += (68<<(j<<3));
        else if(temp < binbounds[15]) temp2 += (64<<(j<<3));
        else if(temp < binbounds[14]) temp2 += (60<<(j<<3));
        else if(temp < binbounds[13]) temp2 += (56<<(j<<3));
        else if(temp < binbounds[12]) temp2 += (52<<(j<<3));
        else if(temp < binbounds[11]) temp2 += (48<<(j<<3));
        else if(temp < binbounds[10]) temp2 += (44<<(j<<3));
        else if(temp < binbounds[9]) temp2 += (40<<(j<<3));
        else if(temp < binbounds[8]) temp2 += (36<<(j<<3));
        else if(temp < binbounds[7]) temp2 += (32<<(j<<3));
        else if(temp < binbounds[6]) temp2 += (28<<(j<<3));
        else if(temp < binbounds[5]) temp2 += (24<<(j<<3));
        else if(temp < binbounds[4]) temp2 += (20<<(j<<3));
        else if(temp < binbounds[3]) temp2 += (16<<(j<<3));
        else if(temp < binbounds[2]) temp2 += (12<<(j<<3));
        else if(temp < binbounds[1]) temp2 += (8<<(j<<3));
        else if(temp < binbounds[0]) temp2 += (4<<(j<<3));
        else temp2 += (0<<(j<<3));
      }
      g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = temp2;
    }
  }
  else {  // blockIdx.x = blockIdx.y, so half the block will be ignorable..
    double temp;
    unsigned int temp2;
    double3 vec1, vec2;

    vec1.x = g_idata1.x[tx];
    vec1.y = g_idata1.y[tx];
    vec1.z = g_idata1.z[tx];
    sdata[threadIdx.x].x = g_idata1.x[by+threadIdx.x];
    sdata[threadIdx.x].y = g_idata1.y[by+threadIdx.x];
    sdata[threadIdx.x].z = g_idata1.z[by+threadIdx.x];

    __syncthreads();

    by <<= (LOG2_GRID_SIZE - 2);
    by += tx;

#pragma unroll
    for(int i=0; i<128; i+=4) {
      temp2 = 0;
#pragma unroll
      for(int j=0; j<4; j++) {
        if(threadIdx.x <= i+j) temp2 += (124<<(j<<3));
        else { 
          vec2 = sdata[i+j];
          temp = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
          if(temp < binbounds[30]) temp2 += (124<<(j<<3));
          else if(temp < binbounds[29]) temp2 += (120<<(j<<3));
          else if(temp < binbounds[28]) temp2 += (116<<(j<<3));
          else if(temp < binbounds[27]) temp2 += (112<<(j<<3));
          else if(temp < binbounds[26]) temp2 += (108<<(j<<3));
          else if(temp < binbounds[25]) temp2 += (104<<(j<<3));
          else if(temp < binbounds[24]) temp2 += (100<<(j<<3));
          else if(temp < binbounds[23]) temp2 += (96<<(j<<3));
          else if(temp < binbounds[22]) temp2 += (92<<(j<<3));
          else if(temp < binbounds[21]) temp2 += (88<<(j<<3));
          else if(temp < binbounds[20]) temp2 += (84<<(j<<3));
          else if(temp < binbounds[19]) temp2 += (80<<(j<<3));
          else if(temp < binbounds[18]) temp2 += (76<<(j<<3));
          else if(temp < binbounds[17]) temp2 += (72<<(j<<3));
          else if(temp < binbounds[16]) temp2 += (68<<(j<<3));
          else if(temp < binbounds[15]) temp2 += (64<<(j<<3));
          else if(temp < binbounds[14]) temp2 += (60<<(j<<3));
          else if(temp < binbounds[13]) temp2 += (56<<(j<<3));
          else if(temp < binbounds[12]) temp2 += (52<<(j<<3));
          else if(temp < binbounds[11]) temp2 += (48<<(j<<3));
          else if(temp < binbounds[10]) temp2 += (44<<(j<<3));
          else if(temp < binbounds[9]) temp2 += (40<<(j<<3));
          else if(temp < binbounds[8]) temp2 += (36<<(j<<3));
          else if(temp < binbounds[7]) temp2 += (32<<(j<<3));
          else if(temp < binbounds[6]) temp2 += (28<<(j<<3));
          else if(temp < binbounds[5]) temp2 += (24<<(j<<3));
          else if(temp < binbounds[4]) temp2 += (20<<(j<<3));
          else if(temp < binbounds[3]) temp2 += (16<<(j<<3));
          else if(temp < binbounds[2]) temp2 += (12<<(j<<3));
          else if(temp < binbounds[1]) temp2 += (8<<(j<<3));
          else if(temp < binbounds[0]) temp2 += (4<<(j<<3));
          else temp2 += (0<<(j<<3));
        }
      }
      g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = temp2;
    }
  }
}


__global__ void ACFKernel(cartesian g_idata1, cartesian g_idata2, unsigned int* g_odata) 
{
  // Shared memory used to store vectors from g_idata2
  extern __shared__ double3 sdata[];
  double temp;
  unsigned int temp2;
  double3 vec1, vec2;
  // tx is the "x position" in the grid
  int tx = (blockIdx.x<<7) + threadIdx.x;
  // "y position" depends on i (see below), this is just y block
  int by = (blockIdx.y<<7);

  // Is coalesced, as cartesians are aligned properly and there are no conflicts.
  vec1.x = g_idata2.x[tx];
  vec1.y = g_idata2.y[tx];
  vec1.z = g_idata2.z[tx];
  // Then reads one unique vector from global to shared per thread, the "shared vectors".
  // Is coalesced for the same reason.
  sdata[threadIdx.x].x = g_idata1.x[by+threadIdx.x];
  sdata[threadIdx.x].y = g_idata1.y[by+threadIdx.x];
  sdata[threadIdx.x].z = g_idata1.z[by+threadIdx.x];
  // Each thread will compute the dot product of its assigned vector with every shared vector.

  // Ensure all reads are finished before using them for any calculations
  __syncthreads();

  // Simplify some notation later on.
  by <<= (LOG2_GRID_SIZE - 2);
  by += tx;

  // Unrolling offers significant speed-up
#pragma unroll
  for(int i=0; i<128; i+=4) {   // Iterate through 128 vectors in sdata
    temp2 = 0;
#pragma unroll
    for(int j=0; j<4; j++) {    // 4 vectors per 1 int output
      // sdata broadcasts sdata[i+j] to all threads in a block; so unnecessary bank conflicts are avoided.
      vec2 = sdata[i+j];
      temp = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
      // This follows the form (binNum << (elementNum << 3)).
      // binNum is the bin we are assigning, elementNum is j, and by summing we pack four bin assignments to one int.
      if(temp < binbounds[30]) temp2 += (124<<(j<<3));
      else if(temp < binbounds[29]) temp2 += (120<<(j<<3));
      else if(temp < binbounds[28]) temp2 += (116<<(j<<3));
      else if(temp < binbounds[27]) temp2 += (112<<(j<<3));
      else if(temp < binbounds[26]) temp2 += (108<<(j<<3));
      else if(temp < binbounds[25]) temp2 += (104<<(j<<3));
      else if(temp < binbounds[24]) temp2 += (100<<(j<<3));
      else if(temp < binbounds[23]) temp2 += (96<<(j<<3));
      else if(temp < binbounds[22]) temp2 += (92<<(j<<3));
      else if(temp < binbounds[21]) temp2 += (88<<(j<<3));
      else if(temp < binbounds[20]) temp2 += (84<<(j<<3));
      else if(temp < binbounds[19]) temp2 += (80<<(j<<3));
      else if(temp < binbounds[18]) temp2 += (76<<(j<<3));
      else if(temp < binbounds[17]) temp2 += (72<<(j<<3));
      else if(temp < binbounds[16]) temp2 += (68<<(j<<3));
      else if(temp < binbounds[15]) temp2 += (64<<(j<<3));
      else if(temp < binbounds[14]) temp2 += (60<<(j<<3));
      else if(temp < binbounds[13]) temp2 += (56<<(j<<3));
      else if(temp < binbounds[12]) temp2 += (52<<(j<<3));
      else if(temp < binbounds[11]) temp2 += (48<<(j<<3));
      else if(temp < binbounds[10]) temp2 += (44<<(j<<3));
      else if(temp < binbounds[9]) temp2 += (40<<(j<<3));
      else if(temp < binbounds[8]) temp2 += (36<<(j<<3));
      else if(temp < binbounds[7]) temp2 += (32<<(j<<3));
      else if(temp < binbounds[6]) temp2 += (28<<(j<<3));
      else if(temp < binbounds[5]) temp2 += (24<<(j<<3));
      else if(temp < binbounds[4]) temp2 += (20<<(j<<3));
      else if(temp < binbounds[3]) temp2 += (16<<(j<<3));
      else if(temp < binbounds[2]) temp2 += (12<<(j<<3));
      else if(temp < binbounds[1]) temp2 += (8<<(j<<3));
      else if(temp < binbounds[0]) temp2 += (4<<(j<<3));
      else temp2 += (0<<(j<<3));
    }
    g_odata[by+(i<<(LOG2_GRID_SIZE - 2))] = temp2;
  }
}

#endif
