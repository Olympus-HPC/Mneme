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

#ifndef _GPU_COMPUTE_H_
#define _GPU_COMPUTE_H_

#include "kernel.h"
#include "ACF_kernel.cu"
#include "histogram_kernel.cu"
#include "model_io.cu"
#include "args.h"
#include <sys/time.h>

#define TDIFF(ts, te) (te.tv_sec - ts.tv_sec + (te.tv_usec - ts.tv_usec) * 1e-6)

#define GRID_SIZE  (1 << LOG2_GRID_SIZE)

const dim3 grid(128, 128, 1);
const dim3 threads(128, 1, 1);

// Device-side data storage
cartesian d_idata1;
cartesian d_idata2;
unsigned int* d_odata1;

// Host-side data storage
cartesian h_idata1;
cartesian h_idata2;

// Performance
struct timeval t1, t0;
float t_Compute = 0.0f;

// Writes bin boundaries to GPU constant memory.
void writeBoundaries(double *binbs) {
  cudaMemcpyToSymbol(binbounds, (void*)binbs, (NUMBINS-1)*sizeof(double));
}

// Used to compute DD or RR, takes advantage of symmetry to reduce number of dot products and waterfall searches
// required by half. Unfortunately, due to the limitations of the histogram kernel, every element of d_odata still
// represents a histogram bin assignment; consequently the histogram kernel does just as much work in tileComputeSymm
// as it does in tileCompute.

// type: type = 0 corresponds to DD, type = 1 corresponds to RR.
// size: Number of elements in data or random set (dependent upon type)
// njk: Number of jackknives
// jkSizes: List of jackknife sizes, in order
// nBins: Number of histogram bins
// histo: The function outputs by adding on to this histogram
// stream: CUDA stream
void tileComputeSymm(int type, int size, int njk, int* jkSizes, int nBins, long long* histo, cudaStream_t &stream) {
  // Storage for GPUHistogram output
  unsigned int* subHistoTemp = (unsigned int*)malloc(njk*nBins*sizeof(unsigned int));

  // Find number of kernels necessary on each 'axis'
  int nkernels = iDivUp(size, GRID_SIZE);

  // Stores location in d_odata; used in GPUHistogram calls
  int index;

  for(int i=0; i<nkernels; i++) {
    if(type == 0) {
      cudaMemcpyAsync(d_idata1.x, &h_idata1.x[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_idata1.y, &h_idata1.y[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_idata1.z, &h_idata1.z[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
    }
    else {
      cudaMemcpyAsync(d_idata1.x, &h_idata2.x[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_idata1.y, &h_idata2.y[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_idata1.z, &h_idata2.z[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
    }
    ACFKernelSymm<<< grid, threads, 128*sizeof(double3), stream >>>(d_idata1, d_odata1);
    index = 0;
    memset(subHistoTemp, 0, njk*nBins*sizeof(unsigned int));
    for(int k=0; k<njk; k++) {
      if(jkSizes[i*njk + k] != 0) {
        GPUHistogram(&subHistoTemp[nBins*k], &d_odata1[index], jkSizes[i*njk + k]*GRID_SIZE, stream);
        index += jkSizes[i*njk + k] * (GRID_SIZE >> 2);
      }
    }
    for(int k=0; k<njk*nBins; k++) {
      histo[k] += subHistoTemp[k];
    }

    for(int j=i+1; j<nkernels; j++) {
      if(type == 0) {
        cudaMemcpyAsync(d_idata2.x, &h_idata1.x[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_idata2.y, &h_idata1.y[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_idata2.z, &h_idata1.z[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      }
      else {
        cudaMemcpyAsync(d_idata2.x, &h_idata2.x[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_idata2.y, &h_idata2.y[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_idata2.z, &h_idata2.z[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      }
      ACFKernel<<< grid, threads, 128*sizeof(double3), stream >>>(d_idata1, d_idata2, d_odata1);
      index = 0;
      memset(subHistoTemp, 0, njk*nBins*sizeof(unsigned int));
      for(int k=0; k<njk; k++) {
        if(jkSizes[i*njk + k] != 0) {
          GPUHistogram(&subHistoTemp[nBins*k], &d_odata1[index], jkSizes[i*njk + k]*GRID_SIZE, stream);
          index += jkSizes[i*njk + k] * (GRID_SIZE >> 2);
        }
      }
      for(int k=0; k<njk*nBins; k++) {
        histo[k] += subHistoTemp[k];
      }
    }
  }
}

// Used to compute DR. 
// dataSize: Size of data set
// randomSize: Size of random set
// All else: See descriptions in tileComputeSymm
void tileCompute(int dataSize, int randomSize, int njk, int* jkSizes, int nBins, long long* histo, cudaStream_t &stream) {
  unsigned int* subHistoTemp = (unsigned int*)malloc(njk*nBins*sizeof(unsigned int));

  int ndkernels = iDivUp(dataSize, GRID_SIZE);
  int nrkernels = iDivUp(randomSize, GRID_SIZE);

  int index;

  for(int i=0; i<ndkernels; i++) {
    cudaMemcpyAsync(d_idata1.x, &h_idata1.x[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_idata1.y, &h_idata1.y[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_idata1.z, &h_idata1.z[i*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
    for(int j=0; j<nrkernels; j++) {
      cudaMemcpyAsync(d_idata2.x, &h_idata2.x[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_idata2.y, &h_idata2.y[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_idata2.z, &h_idata2.z[j*GRID_SIZE], GRID_SIZE*sizeof(double), cudaMemcpyHostToDevice, stream);
      ACFKernel<<< grid, threads, 128*sizeof(double3), stream >>>(d_idata1, d_idata2, d_odata1);
      index = 0;
      memset(subHistoTemp, 0, njk*nBins*sizeof(unsigned int));
      for(int k=0; k<njk; k++) {
        if(jkSizes[i*njk + k] != 0) {
          GPUHistogram(&subHistoTemp[nBins*k], &d_odata1[index], jkSizes[i*njk + k]*GRID_SIZE, stream);
          index += jkSizes[i*njk + k] * (GRID_SIZE >> 2);
        }
      }
      for(int k=0; k<njk*nBins; k++) {
        histo[k] += subHistoTemp[k];
      }
    }
  }
}


// Computes histograms and writes to DDs, DRs, RRs. These must be compiled by the host program;
// the function outputs njk sub-histograms for each, the sum of which is the full histogram.
// Note that data should be sorted according to jackknife; otherwise results will be meaningless.

// dataName: File name of data points file.
// randomNames: File name stem of random points file.
// nr: Number of random files. Random files are assumed to be of the form randomNames.i where 1 <= i <= nr.
// dataSize: Number of elements to read from data points file.
// randomSize: Number of elements to read from each random points file.
// njk: Number of jackknives.
// jkSizes: Ordered list of jackknife sizes. Each size must be a multiple of 4 currently.
// nBins: Number of histogram bins.
// zeroBin: Index of bin which contains 0.0f: Necessary to correct for padding
// DDs, DRs, RRs: Output subhistogram lists.
void doComputeGPU(char* dataName, char* randomNames, int nr, int dataSize, int randomSize, int njk, int* jkSizes, 
    int nBins, int zeroBin, long long** DDs, long long** DRs, long long** RRs) {
  // DDs, DRs, RRs are not assumed to be allocated or cleared.
  *DDs = (long long*)malloc(nBins*njk*sizeof(long long));
  *DRs = (long long*)malloc(nBins*njk*nr*sizeof(long long));
  *RRs = (long long*)malloc(nBins*nr*sizeof(long long));
  memset(*DDs, 0, nBins*njk*sizeof(long long));
  memset(*DRs, 0, nBins*njk*nr*sizeof(long long));
  memset(*RRs, 0, nBins*nr*sizeof(long long));

  int ndkernels = iDivUp(dataSize, GRID_SIZE);
  int nrkernels = iDivUp(randomSize, GRID_SIZE);
  int* dkerneljkSizes = (int*)malloc(njk*ndkernels*sizeof(int));
  int* rkerneljkSizes = (int*)malloc(njk*nrkernels*sizeof(int));
  memset(dkerneljkSizes, 0, njk*ndkernels*sizeof(int));
  memset(rkerneljkSizes, 0, njk*nrkernels*sizeof(int));

  int currentjk = 0;
  int numwrittencurrentjk = 0;
  for(int i=0; i<ndkernels; i++) {
    int remainder = GRID_SIZE;
    while(remainder > 0 && currentjk < njk) {
      if(remainder < jkSizes[currentjk] - numwrittencurrentjk) {
        dkerneljkSizes[i*njk + currentjk] += remainder;
        numwrittencurrentjk += remainder;
        remainder = 0;
      }
      else {
        remainder = remainder - (jkSizes[currentjk] - numwrittencurrentjk);
        dkerneljkSizes[i*njk + currentjk] += (jkSizes[currentjk] - numwrittencurrentjk);
        currentjk++;
        numwrittencurrentjk = 0;
      }
    }
  }

  for(int i=0; i<nrkernels-1; i++) {
    rkerneljkSizes[i*njk] += GRID_SIZE;
  }
  rkerneljkSizes[(nrkernels-1)*njk] += (randomSize % GRID_SIZE == 0) ? GRID_SIZE : randomSize % GRID_SIZE;

  // Kernel invocations require that the input data have 16384 elements, so pad the data to a multiple of 16384.
  int dataSizePadded = dataSize + ((GRID_SIZE - (dataSize % GRID_SIZE)) % GRID_SIZE);
  int randomSizePadded = randomSize + ((GRID_SIZE - (randomSize % GRID_SIZE)) % GRID_SIZE);

  // Use page-locked host memory; somewhat faster, and host-side memory requirements are rather small.
  cudaMallocHost((void**)&h_idata1.x, dataSizePadded*sizeof(double));
  cudaMallocHost((void**)&h_idata1.y, dataSizePadded*sizeof(double));
  cudaMallocHost((void**)&h_idata1.z, dataSizePadded*sizeof(double));
  h_idata1.jk = (int*)malloc(dataSize*sizeof(int));

  cudaMallocHost((void**)&h_idata2.x, randomSizePadded*sizeof(double));
  cudaMallocHost((void**)&h_idata2.y, randomSizePadded*sizeof(double));
  cudaMallocHost((void**)&h_idata2.z, randomSizePadded*sizeof(double));
  h_idata2.jk = (int*)malloc(randomSize*sizeof(int));

  // Ensure that the dot product of a padding element and any other element gets mapped to the bin containing 0.0f.
  for(int i=dataSize; i<dataSizePadded; i++) {
    h_idata1.x[i] = double(0.0);
    h_idata1.y[i] = double(0.0);
    h_idata1.z[i] = double(0.0);
  }
  for(int i=randomSize; i<randomSizePadded; i++) {
    h_idata2.x[i] = double(0.0);
    h_idata2.y[i] = double(0.0);
    h_idata2.z[i] = double(0.0);
  }

  // Allocate device memory for inputs and output.
  cudaMalloc((void**)&d_idata1.x, GRID_SIZE*sizeof(double));
  cudaMalloc((void**)&d_idata1.y, GRID_SIZE*sizeof(double));
  cudaMalloc((void**)&d_idata1.z, GRID_SIZE*sizeof(double));
  d_idata1.jk = NULL;
  cudaMalloc((void**)&d_idata2.x, GRID_SIZE*sizeof(double));
  cudaMalloc((void**)&d_idata2.y, GRID_SIZE*sizeof(double));
  cudaMalloc((void**)&d_idata2.z, GRID_SIZE*sizeof(double));
  d_idata2.jk = NULL;
  cudaMalloc((void**)&d_odata1, GRID_SIZE*GRID_SIZE*sizeof(unsigned int)/4);

  histoInit();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  struct timeval t3, t2, t1, t0;
  float t_computeDD=0, t_computeRRS=0, t_computeDRS=0, t_fileIO=0;

  gettimeofday(&t0, NULL);

  char fname[256];
  readdatafile(dataName, h_idata1, dataSize);

  gettimeofday(&t1, NULL);

  // Compute DD
  tileComputeSymm(0, dataSize, njk, dkerneljkSizes, nBins, *DDs, stream);

  gettimeofday(&t2, NULL);

  t_fileIO += TDIFF(t0, t1);
  t_computeDD = TDIFF(t1, t2);

  for(int i=0; i<nr; i++) {
    sprintf(fname, "%s.%i", randomNames, i+1);
    gettimeofday(&t0, NULL);

    readdatafile(fname, h_idata2, randomSize);
    gettimeofday(&t1, NULL);

    // Compute DR_i
    tileCompute(dataSize, randomSize, njk, dkerneljkSizes, nBins, &(*DRs)[njk*nBins*i], stream);
    gettimeofday(&t2, NULL);

    // Compute RR_i
    tileComputeSymm(1, randomSize, njk, rkerneljkSizes, nBins, &(*RRs)[nBins*i], stream);
    gettimeofday(&t3, NULL);

    t_fileIO += TDIFF(t0, t1);  
    t_computeDRS += TDIFF(t1, t2);
    t_computeRRS += TDIFF(t2, t3);
  }

  // Correct for error introduced by padding vectors.
  int padfactor = randomSizePadded - randomSize;
  for(int i=0; i<nr; i++) {
    (*RRs)[nBins*i + zeroBin] -= padfactor*randomSize;
    for(int j=0; j<njk; j++) {
      (*DRs)[nBins*njk*i + nBins*j + zeroBin] -= padfactor*jkSizes[j];
    }
  }
  for(int i=0; i<njk; i++) {
    (*DDs)[nBins*i + zeroBin] -= padfactor*jkSizes[i];
  }

  // Tidy up.
  cudaStreamDestroy(stream);
  histoClose();
  cudaFree(d_idata1.x);
  cudaFree(d_idata1.y);
  cudaFree(d_idata1.z);
  cudaFree(d_idata2.x);
  cudaFree(d_idata2.y);
  cudaFree(d_idata2.z);
  cudaFree(d_odata1);
  cudaFreeHost(h_idata1.x);
  cudaFreeHost(h_idata1.y);
  cudaFreeHost(h_idata1.z);
  free(h_idata1.jk);
  cudaFreeHost(h_idata2.x);
  cudaFreeHost(h_idata2.y);
  cudaFreeHost(h_idata2.z);
  free(h_idata2.jk);

  printf("================================================\n");
  printf("Time to compute DD: %.4f sec\n", t_computeDD);
  printf("Time to compute RRS: %.4f sec\n", t_computeRRS);
  printf("Time to compute DRS: %.4f sec\n", t_computeDRS);
  printf("Time to load data files: %.4f sec\n", t_fileIO);
  printf("Time to compute DD, RRS, & DRS: %.4f sec\n", t_computeDD+t_computeRRS+t_computeDRS);
  printf("TOTAL time (DD+RRS+DRS+IO): %.4f sec\n", t_computeDD+t_computeRRS+t_computeDRS+t_fileIO);
  printf("================================================\n");  
}

void compileHistograms(long long* DDs, long long* DRs, long long* RRs, long long*** DD, long long*** DR, 
    long long*** RR, options *args) {
  *DD = (long long**)malloc(((*args).njk+1)*sizeof(long long*));
  *DR = (long long**)malloc(((*args).njk+1)*sizeof(long long*));
  *RR = (long long**)malloc(1*sizeof(long long*));
  for(int i=0; i<=(*args).njk; i++) {
    (*DD)[i] = (long long*)malloc(NUMBINS*sizeof(long long));
    (*DR)[i] = (long long*)malloc(NUMBINS*sizeof(long long));
    memset((*DD)[i], 0, NUMBINS*sizeof(long long));
    memset((*DR)[i], 0, NUMBINS*sizeof(long long));
  }
  (*RR)[0] = (long long*)malloc(NUMBINS*sizeof(long long));
  memset((*RR)[0], 0, NUMBINS*sizeof(long long));

  for(int i=0; i<NUMBINS; i++) {
    for(int k=0; k<(*args).njk; k++) {
      (*DD)[0][i] += DDs[NUMBINS*k + i];
    }
    for(int j=0; j<(*args).random_count; j++) {
      for(int k=0; k<(*args).njk; k++) {
        (*DR)[0][i] += DRs[(j*(*args).njk + k)*NUMBINS + i];
      }
      (*RR)[0][i] += RRs[j*NUMBINS + i];
    }
  }

  for(int k=1; k<=(*args).njk; k++) {
    for(int i=0; i<NUMBINS; i++) {
      (*DD)[k][i] = (*DD)[0][i] - DDs[(k-1)*NUMBINS + i];
      (*DR)[k][i] = (*DR)[0][i];
      for(int j=0; j<(*args).random_count; j++) {
        (*DR)[k][i] -= DRs[(j*(*args).njk + k - 1)*NUMBINS + i];
      }
    }
  }
}  

#endif



