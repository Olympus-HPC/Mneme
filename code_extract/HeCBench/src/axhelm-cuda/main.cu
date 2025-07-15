#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <cuda.h>

#define POLYNOMIAL_DEGREE  7
#define p_Nggeo 7
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_GWJID 0

#include "meshBasis.hpp"

// cpu reference 
#include "axhelmReference.cpp"

// gpu kernel
#include "axhelmKernel.cpp"

//=========================================================================
// End of kernel
//=========================================================================

dfloat *drandAlloc(int Nelem){

  dfloat *v = (dfloat*) calloc(Nelem, sizeof(dfloat));

  for(int n=0;n<Nelem;++n){
    v[n] = drand48();
  }

  return v;
}

int main(int argc, char **argv){

  if (argc<4) {
    printf("Usage: ./axhelm Ndim numElements [nRepetitions]\n");
    return 1;
  }

  const int Ndim = atoi(argv[1]);
  const int Nelements = atoi(argv[2]);
  int Ntests = 1;
  if(argc>=4)
    Ntests = atoi(argv[3]);

  const int Nq = POLYNOMIAL_DEGREE + 1;
  const int Np = Nq*Nq*Nq;
  const int offset = Nelements*Np;


  // build element nodes and operators
  dfloat *rV, *wV, *DrV;
  meshJacobiGQ(0,0,POLYNOMIAL_DEGREE, &rV, &wV);
  meshDmatrix1D(POLYNOMIAL_DEGREE, Nq, rV, &DrV);

  std::cout << "word size: " << sizeof(dfloat) << " bytes\n";

  // populate device arrays
  dfloat *ggeo = drandAlloc(Np*Nelements*p_Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  const dfloat lambda1 = 1.1;
  dfloat *lambda = (dfloat*) calloc(2*offset, sizeof(dfloat));
  for(int i=0; i<offset; i++) {
    lambda[i]        = 1.0;
    lambda[i+offset] = lambda1;
  }

  // compute the reference result
  for(int n=0;n<Ndim;++n){
    dfloat *x = q + n*offset;
    dfloat *Ax = Aq + n*offset; 
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }

  auto start = std::chrono::high_resolution_clock::now();

  dfloat *o_ggeo, *o_q, *o_Aq, *o_DrV, *o_lambda;
  cudaMalloc((void**) &o_ggeo, Np*Nelements*p_Nggeo*sizeof(dfloat));
  cudaMalloc((void**) &o_q   , Ndim*Np*Nelements*sizeof(dfloat));
  cudaMalloc((void**) &o_Aq  , Ndim*Np*Nelements*sizeof(dfloat));
  cudaMalloc((void**) &o_DrV , Nq*Nq*sizeof(dfloat));
  cudaMalloc((void**) &o_lambda , 2*offset*sizeof(dfloat));

  cudaMemcpy(o_ggeo, ggeo, Np*Nelements*p_Nggeo*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_q, q, (Ndim*Np)*Nelements*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_DrV, DrV, Nq*Nq*sizeof(dfloat), cudaMemcpyHostToDevice);
  cudaMemcpy(o_lambda, lambda, 2*offset*sizeof(dfloat), cudaMemcpyHostToDevice);

  for(int test=0;test<Ntests;++test) {
    if (Ndim > 1)
      axhelm_n3 <<< dim3(Nelements,1), dim3(8,8) >>> (
          Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq);
    else
      axhelm <<< dim3(Nelements,1), dim3(8,8) >>> (
          Nelements, offset, o_ggeo, o_DrV, o_lambda, o_q, o_Aq);
  }

  // store the device results in the 'q' array
  cudaMemcpy(q, o_Aq, Ndim*Np*Nelements*sizeof(dfloat), cudaMemcpyDeviceToHost);

  cudaFree(o_ggeo);
  cudaFree(o_q);
  cudaFree(o_Aq);
  cudaFree(o_DrV);
  cudaFree(o_lambda);

  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / Ntests;

  // verify
  dfloat maxDiff = 0;
  for(int n=0;n<Ndim*Np*Nelements;++n){
    dfloat diff = fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  std::cout << "Correctness check: maxError = " << maxDiff << "\n";
  free(ggeo);
  free(q);
  free(Aq);
  free(lambda);
  free(rV);
  free(wV);
  free(DrV);

  // print statistics
  const dfloat GDOFPerSecond = Ndim*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*POLYNOMIAL_DEGREE*Nelements/elapsed;
  const long long bytesMoved = (Ndim*2*Np+7*Np+2*Np)*sizeof(dfloat); // x, Mx, opa, lambda
  const double bw = bytesMoved*Nelements/elapsed;
  double flopCount = Ndim*Np*12*Nq;
  if(Ndim == 1) flopCount += 22*Np;
  if(Ndim == 3) flopCount += 69*Np;
  double gflops = flopCount*Nelements/elapsed;
  std::cout << " NRepetitions=" << Ntests
    << " Ndim=" << Ndim
    << " N=" << POLYNOMIAL_DEGREE
    << " Nelements=" << Nelements
    << " elapsed time=" << elapsed
    << " GDOF/s=" << GDOFPerSecond
    << " GB/s=" << bw
    << " GFLOPS/s=" << gflops
    << "\n";
  return 0;
}
