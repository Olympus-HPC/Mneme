#include <iostream>
#include "mttkrp_gpu.h"
#include <vector>

inline cudaError_t checkCuda(cudaError_t result, int s){

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error at line %d: %s\n", s, cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2, 
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2, 
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW)
{

  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = (gId >> (5 + logOfWPC)) << logOfFPW; // 5: minimum 1 WARP (2^5) // blockIdx.x ;//

  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){       

    tmp_val = 0;
    bool diffFiber = false;
    unsigned int idx0;

    for (int fr = 0; fr < fbrPerWarp && (fbr+fr) < (nFibers - 1); ++fr){

      diffFiber = false;
      unsigned int idx1 = fbrIdx1[fbr+fr];// dInds1[fbrPtr1[fbr]];  
      idx0 = fbrLikeSlcInds[fbr+fr];//slc;  
      tmp_val = 0;

      for(unsigned int x = fbrPtr1[fbr+fr] + workId; x < fbrPtr1[fbr+fr+1]; x+=warpPerSlice) {

        unsigned int idx2 = dInds2[x];                    

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp_val += vals[x] * dU2[idx2 * R + r]; //2MR   
        }       
      }

      for(unsigned int r=laneId; r<R; r+=32) { 
        tmp += tmp_val * dU1[idx1 * R + r] ; //2PR
      } 

      if(fbrLikeSlcInds[fbr+fr] != fbrLikeSlcInds[fbr+fr+1]) {

        diffFiber = true;
        for(unsigned int r=laneId; r<R; r+=32) { 
          atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
        } 
        tmp = 0;
      }
    } 

    if(!diffFiber) {  
      for(unsigned int r=laneId; r<R; r+=32) { 
        atomicAdd(&dU0[idx0 * R + r], tmp); 
      }  
    }  
  }
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3, 
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0, 
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE  mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC,
  int fbrPerWarp,
  int logOfFPW)
{
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = (gId >> (5 + logOfWPC)) << logOfFPW; // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val, tmp2= 0;

  if(fbrS < nFibers - 1){       

    tmp_val = 0;
    bool diffFiber = false;
    unsigned int idx0;

    for (int fr = 0; fr < fbrPerWarp && (fbrS+fr) < (nFibers - 1); ++fr){

      diffFiber = false;
      unsigned int idx1 = fbrIdx1[fbrS+fr];// dInds1[fbrPtr1[fbr]];  
      idx0 = fbrLikeSlcInds[fbrS+fr];//slc;  
      tmp = 0;

      for (int fbr = fbrPtr1[fbrS+fr] + workId; fbr < fbrPtr1[fbrS+fr+1]; fbr+=warpPerSlice){
        ITYPE idx2 = fbrIdx2[fbr];
        tmp_val = 0;

        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; x++) {

          unsigned int idx3 = dInds3[x];

          for(unsigned int r=laneId; r<R; r+=32) {
            tmp_val += vals[x] * dU3[idx3 * R + r]; //2MR   
          }       
        }

        for(unsigned int r=laneId; r<R; r+=32) { 
          tmp += tmp_val * dU2[idx2 * R + r] ;
        } 
      }
      for(unsigned int r=laneId; r<R; r+=32) { 
        tmp2 += tmp * dU1[idx1 * R + r] ;
      } 

      if(fbrLikeSlcInds[fbrS+fr] != fbrLikeSlcInds[fbrS+fr+1]) {

        diffFiber = true;
        for(unsigned int r=laneId; r<R; r+=32) { 
          atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR
        } 
        tmp2 = 0;
      }
    }

    if(!diffFiber) {  
      for(unsigned int r=laneId; r<R; r+=32) 
        atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR           
    }  
  }
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3, 
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
  DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC)
{

  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val, tmp2 = 0;

  if(fbrS < nFibers - 1){       

    tmp = 0;
    unsigned int idx0 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];  
    unsigned int idx3 = fbrLikeSlcInds[fbrS];//slc;  

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      unsigned int idx1 = fbrIdx2[fbr];
      tmp_val = 0;

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx2 = dInds3[x];                    

        for(unsigned int r=laneId; r<R; r+=32) 
          tmp_val += vals[x] * dU2[idx2 * R + r] ; //2MR
      }
      for(unsigned int r=laneId; r<R; r+=32)  
        tmp += tmp_val * dU1[idx1 * R + r]  ;  
    }     
    for(unsigned int r=laneId; r<R; r+=32) { 
      tmp2 = tmp * dU3[idx3 * R + r];
      atomicAdd(&dU0[idx0 * R + r], tmp2); //2PR
    }    
  }
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2, 
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2, 
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC)
{
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){       

    tmp_val = 0;
    unsigned int idx0 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];  
    unsigned int idx2 = fbrLikeSlcInds[fbr];//slc; 

    for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

      unsigned int idx1 = dInds2[x];                    

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp_val += vals[x] * dU1[idx1 * R + r]; //2MR
      }
    }     
    for(unsigned int r=laneId; r<R; r+=32) { 
      tmp = tmp_val * dU2[idx2 * R + r] ;
      atomicAdd(&dU0[idx0 * R + r], tmp); //2PR
    }    
  }
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3, 
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC)
{
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp;

  if(fbrS < nFibers - 1){       

    unsigned int idx2 = fbrLikeSlcInds[fbrS];//slc;  
    unsigned int idx3 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];  

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      unsigned int idx0 = fbrIdx2[fbr];
      tmp = 0;

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx1 = dInds3[x];                    

        for(unsigned int r=laneId; r<R; r+=32) 
          tmp += vals[x] * dU1[idx1 * R + r]; //2MR
      }
      for(unsigned int r=laneId; r<R; r+=32)  {
        atomicAdd(&dU0[idx0 * R + r], tmp * dU2[idx2 * R + r] * dU3[idx3 * R + r]) ;  
      }
    }            
  }
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds2, 
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2, 
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC)
{
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbr = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val;

  if(fbr < nFibers - 1){       

    tmp_val = 0;
    unsigned int idx1 = fbrLikeSlcInds[fbr];//slc;  
    unsigned int idx2 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];  

    for(unsigned int r=laneId; r<R; r+=32) 
      tmp = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR

    for(unsigned int x = fbrPtr1[fbr] + workId; x < fbrPtr1[fbr+1]; x+=warpPerSlice) {

      unsigned int idx0 = dInds2[x];                    

      for(unsigned int r=laneId; r<R; r+=32) {
        tmp_val = vals[x] * tmp;///dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //2MR
        atomicAdd(&dU0[idx0 * R + r], tmp_val);
      }
    }         
  }
}

// CUDA fbr atomic sing slcLikeFbr
__global__ void mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D(
  const DTYPE *__restrict__ vals,
  const ITYPE *__restrict__ fbrLikeSlcInds,
  const ITYPE *__restrict__ dInds3, 
  const ITYPE *__restrict__ fbrPtr0,
  const ITYPE *__restrict__ fbrPtr1,
  const ITYPE *__restrict__ fbrIdx1,
  const ITYPE *__restrict__ fbrPtr2,
  const ITYPE *__restrict__ fbrIdx2,
  ITYPE nFibers,
        DTYPE *__restrict__ dU0,
  const DTYPE *__restrict__ dU1,
  const DTYPE *__restrict__ dU2,
  const DTYPE *__restrict__ dU3,
  ITYPE mode,
  ITYPE R,
  ITYPE warpPerSlice,
  int logOfWPC)
{
  ITYPE tId = threadIdx.x;
  ITYPE laneId = tId & 31;
  ITYPE bdim = blockDim.x;
  ITYPE gId = (blockIdx.x * bdim + tId);
  ITYPE workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;  //tId >> 5; //tId >> 5;//
  ITYPE fbrS = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5) // blockIdx.x ;//
  DTYPE tmp = 0, tmp_val = 0;;

  if(fbrS < nFibers - 1){       

    tmp = 0;
    unsigned int idx1 = fbrLikeSlcInds[fbrS];//slc;  
    unsigned int idx2 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];                

    for(unsigned int r=laneId; r<R; r+=32) 
      tmp_val = dU1[idx1 * R + r] * dU2[idx2 * R + r] ; //1PR

    for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
      ITYPE idx3 = fbrIdx2[fbr];

      for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {
        unsigned int idx0 = dInds3[x];  

        for(unsigned int r=laneId; r<R; r+=32) {
          tmp = vals[x] * dU3[idx3 * R + r] * tmp_val;//2MR
          atomicAdd(&dU0[idx0 * R + r], tmp);
        }
      }
    }            
  }
}


int MTTKRP_MIHCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){

  ITYPE *dInds2, *dInds3, *dFbrPtr0, *dFbrIdx0, *dfbrPtr1, *dFbrIdx1, *dFbrPtr2, *dFbrIdx2, *dFbrLikeSlcInds;
  DTYPE *dVals;
  ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0,  dFbrIdxLoc =0, dFbrLoc2 =0;
  ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0, totFbrIdx = 0, totFbrPtr2 = 0;

  // All m same mode
  ITYPE mode0 = 0;//TiledX[0].modeOrder[0];
  ITYPE mode1 = 1;//TiledX[0].modeOrder[1];
  ITYPE mode2 = 2;//TiledX[0].modeOrder[2];
  ITYPE mode3 = 3;//((TiledX[0].ndims == 4) ? TiledX[0].modeOrder[3] : 0) ;

  for (int m = 0; m < TiledX[0].ndims; ++m){

    if (TiledX[m].totNnz == 0) continue;

    totNnz += TiledX[m].totNnz;
    totSlcPtr += TiledX[m].fbrPtr[0].size() ;
    totSlcIdx += TiledX[m].fbrIdx[0].size() ;
    totFbrPtr += TiledX[m].fbrPtr[1].size() ;
    totFbrIdx += TiledX[m].fbrIdx[1].size() ;
    totFbrPtr2 += ((TiledX[m].ndims == 4) ? TiledX[m].fbrPtr[2].size() : 0) ;
  }

  // Allocate Tensor on a device
  checkCuda(cudaMalloc((void**) &dVals, totNnz * sizeof(DTYPE)), __LINE__);
  checkCuda(cudaMalloc((void**) &dFbrPtr0, totSlcPtr * sizeof(ITYPE)), __LINE__);
  checkCuda(cudaMalloc((void**) &dFbrIdx0, totSlcIdx * sizeof(ITYPE)), __LINE__);
  checkCuda(cudaMalloc((void**) &dfbrPtr1, totFbrPtr * sizeof(ITYPE)), __LINE__);
  checkCuda(cudaMalloc((void**) &dFbrIdx1, totFbrIdx * sizeof(ITYPE)), __LINE__);
  checkCuda(cudaMalloc((void**) &dFbrLikeSlcInds, totFbrIdx * sizeof(ITYPE)), __LINE__);

  if(TiledX[0].ndims == 3)
    checkCuda(cudaMalloc((void**) &dInds2, totNnz * sizeof(ITYPE)), __LINE__);

  if(TiledX[0].ndims == 4){
    checkCuda(cudaMalloc((void**) &dFbrIdx2, totFbrPtr2 * sizeof(ITYPE)), __LINE__);
    checkCuda(cudaMalloc((void**) &dFbrPtr2, totFbrPtr2 * sizeof(ITYPE)), __LINE__);
    checkCuda(cudaMalloc((void**) &dInds3, totNnz * sizeof(ITYPE)), __LINE__);
  }

  // device memory copy for tiled parts
  for (int m = 0; m < TiledX[0].ndims; ++m){  

    if(m > 0) {

      if (TiledX[m-1].totNnz > 0) {

        dLoc += TiledX[m-1].totNnz;
        dSlcLoc += TiledX[m - 1].fbrPtr[0].size(); // all m same
        dSlcIdxLoc += TiledX[m - 1].fbrIdx[0].size(); 
        dFbrLoc += TiledX[m - 1].fbrPtr[1].size();
        dFbrIdxLoc += TiledX[m - 1].fbrIdx[1].size();
        dFbrLoc2 += ((TiledX[m].ndims == 4) ? TiledX[m - 1].fbrPtr[2].size() : 0) ;
      }
    }

    if (TiledX[m].totNnz == 0) continue;

    checkCuda(cudaMemcpyAsync(dVals + dLoc, &(TiledX[m].vals[0]), 
          TiledX[m].totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice, 0), __LINE__);
    checkCuda(cudaMemcpyAsync(dFbrPtr0 + dSlcLoc, &(TiledX[m].fbrPtr[0][0]), 
          TiledX[m].fbrPtr[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);
    checkCuda(cudaMemcpyAsync(dFbrIdx0 + dSlcIdxLoc, &(TiledX[m].fbrIdx[0][0]), 
          TiledX[m].fbrIdx[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);
    checkCuda(cudaMemcpyAsync(dfbrPtr1 + dFbrLoc, &(TiledX[m].fbrPtr[1][0]), 
          TiledX[m].fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);
    checkCuda(cudaMemcpyAsync(dFbrIdx1 + dFbrIdxLoc, &(TiledX[m].fbrIdx[1][0]), 
          TiledX[m].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);
    checkCuda(cudaMemcpyAsync(dFbrLikeSlcInds + dFbrIdxLoc, &(TiledX[m].fbrLikeSlcInds[0]), 
          TiledX[m].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);

    if(TiledX[m].ndims == 3){
      if(m <= 2)
        checkCuda(cudaMemcpyAsync(dInds2 + dLoc, &(TiledX[m].inds[TiledX[m].modeOrder[2]][0]), 
              TiledX[m].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);      
    }
    if(TiledX[m].ndims == 4){      
      checkCuda(cudaMemcpyAsync(dFbrPtr2 + dFbrLoc2, &(TiledX[m].fbrPtr[2][0]),
            TiledX[m].fbrPtr[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);
      checkCuda(cudaMemcpyAsync(dFbrIdx2 + dFbrLoc2, &(TiledX[m].fbrIdx[2][0]),
            TiledX[m].fbrIdx[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);
      checkCuda(cudaMemcpyAsync(dInds3 + dLoc, &(TiledX[m].inds[TiledX[m].modeOrder[3]][0]),
            TiledX[m].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice, 0), __LINE__);
    }
  }

  //Matrices
  unsigned int *dULoc =  new unsigned int[TiledX[0].ndims];
  unsigned int *szDU =  new unsigned int[TiledX[0].ndims];

  //Matrices
  DTYPE *dU;

  ITYPE mtxSize = ((TiledX[0].ndims == 3) ? (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows) * U[mode0].nCols
      : (U[mode0].nRows + U[mode1].nRows + U[mode2].nRows + U[mode3].nRows) * U[mode0].nCols );

  checkCuda(cudaMalloc((void**) &dU, mtxSize * sizeof(DTYPE)), __LINE__);

  for (int m = 0; m < TiledX[0].ndims; ++m)
    szDU[m] = U[m].nRows * U[m].nCols;

  cudaMemset(dU, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
  checkCuda(cudaMemcpyAsync(dU + szDU[0], &(U[mode1].vals[0]), 
        U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice, 0), __LINE__);
  checkCuda(cudaMemcpyAsync(dU + szDU[0] + szDU[1], &(U[mode2].vals[0]),
        U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice, 0), __LINE__);

  if(TiledX[0].ndims == 4)
    checkCuda(cudaMemcpyAsync(dU + szDU[0] + szDU[1] + szDU[2], &(U[mode3].vals[0]), 
        U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice, 0), __LINE__);

  dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0, dFbrIdxLoc = 0, dFbrLoc2= 0;

  for (int MTTKRPmode = 0; MTTKRPmode < TiledX[0].ndims; ++MTTKRPmode){

    if(MTTKRPmode > 0){

      dLoc = 0; dSlcLoc = 0; dSlcIdxLoc = 0; dFbrLoc =0; dFbrIdxLoc = 0; dFbrLoc2= 0;

      // MTTKRP on mode mode 0 changed DU0. To pass correctness for now initializing to 2 again.
      int mode = MTTKRPmode - 1;
      for(long r = 0; r < U[mode].nRows; ++r){
        for(long c = 0; c < U[mode].nCols; ++c)
          U[mode].vals[r * U[mode].nCols + c] = mode + .5;
      }

      if(MTTKRPmode == 1){
        checkCuda(cudaMemcpyAsync(dU, &(U[mode0].vals[0]), 
              U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice, 0), __LINE__);  
        cudaMemset(dU + szDU[0], 0,  U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE));
      }
      else if(MTTKRPmode == 2){
        checkCuda(cudaMemcpyAsync(dU + szDU[0], &(U[mode1].vals[0]), 
              U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice, 0), __LINE__);  
        cudaMemset(dU + szDU[0] + szDU[1], 0,  U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE));
      }
      else if(MTTKRPmode == 3){
        checkCuda(cudaMemcpyAsync(dU + szDU[0] + szDU[1] , &(U[mode2].vals[0]), 
              U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice, 0), __LINE__);  
        cudaMemset(dU + szDU[0] + szDU[1] + szDU[2], 0,  U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE));
      }
    }

    for (int m = 0; m < TiledX[0].ndims; ++m){

      /* matrix order according to mode order*/ 
      for (int mm = 0; mm < TiledX[0].ndims; ++mm){

        int curMode = TiledX[m].modeOrder[mm];
        dULoc[mm] = 0;

        for (int q = 0; q < curMode; ++q)
          dULoc[mm] +=  szDU[q % TiledX[0].ndims]; //1 2 3 0
      }  

      if(m > 0) {

        if (TiledX[m-1].totNnz > 0) {

          dLoc += TiledX[m-1].totNnz;
          dSlcLoc += TiledX[m - 1].fbrPtr[0].size(); 
          dSlcIdxLoc += TiledX[m - 1].fbrIdx[0].size(); 
          dFbrLoc += TiledX[m - 1].fbrPtr[1].size();
          dFbrIdxLoc += TiledX[m - 1].fbrIdx[1].size();
          dFbrLoc2 += ((TiledX[0].ndims == 4) ? TiledX[m - 1].fbrPtr[2].size(): 0) ;
        }
      }

      if (TiledX[m].totNnz == 0) continue;

      int BLOCKSIZE;

      if(TiledX[m].modeOrder[0] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: Slc atomics\n" ;

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice;//4;
        int logOfWarpPerFbr = log2(warpPerFbr);
        int fbrPerWarp = Opt.fiberPerWarp;//1;//BLOCKSIZE/32; // dont overflow TB
        int logOfFbrPerWarp = log2(fbrPerWarp );

        if( (warpPerFbr > (BLOCKSIZE/32)) || (fbrPerWarp > (BLOCKSIZE/32)) ){
          std::cout << "warpPerFbr (-w) or fbrPerWarp (-s) cannot be higher than threadblock size!"
            << std::endl << "hint: increase -b!" << std::endl;
          return -1;
        }

        grid.x = ( warpPerFbr * 32 * ((TiledX[m].nFibers + fbrPerWarp-1)/fbrPerWarp) + BLOCKSIZE - 1) / BLOCKSIZE;

        if(TiledX[0].ndims == 3)
          mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar<<<grid, block, 0, 0>>>(dVals + dLoc, dFbrLikeSlcInds + dFbrIdxLoc, 
              dInds2 + dLoc, dFbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dFbrIdx1 + dFbrIdxLoc, TiledX[m].nFibers, 
              dU + dULoc[0], dU + dULoc[1], dU + dULoc[2], Opt.mode, Opt.R, warpPerFbr, logOfWarpPerFbr, fbrPerWarp, logOfFbrPerWarp);

        else if(TiledX[0].ndims == 4) {
          mttkrp_MIHCSR_kernel_slc_atomic_fbrLvlPar_4D<<<grid, block, 0, 0>>>(dVals + dLoc, dFbrLikeSlcInds + dFbrIdxLoc, 
              dInds3 + dLoc, dFbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dFbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, 
              TiledX[m].nFibers, dU + dULoc[0], dU + dULoc[1], dU + dULoc[2], dU + dULoc[3], Opt.mode, Opt.R, warpPerFbr, logOfWarpPerFbr, fbrPerWarp, logOfFbrPerWarp);
	}
      }

      else if(TiledX[0].ndims == 4 && TiledX[m].modeOrder[1] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: FbrS atomics\n";

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        grid.x = ( warpPerFbr * 32 * TiledX[m].nFibers + BLOCKSIZE - 1) / BLOCKSIZE;

        mttkrp_MIHCSR_kernel_fbrS_atomic_fbrLvlPar_4D<<<grid, block, 0, 0>>>(dVals + dLoc, dFbrLikeSlcInds + dFbrIdxLoc, 
            dInds3 + dLoc, dFbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dFbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, 
            TiledX[m].nFibers, dU + dULoc[1], dU + dULoc[2], dU + dULoc[3], dU + dULoc[0], Opt.mode, Opt.R, warpPerFbr, logOfWarpPerFbr);
      }

      else if(TiledX[m].modeOrder[TiledX[0].ndims-2] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: Fbr atomics\n";

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        grid.x = ( warpPerFbr * 32 * TiledX[m].nFibers + BLOCKSIZE - 1) / BLOCKSIZE;

        if(TiledX[0].ndims == 3)
          mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar<<<grid, block, 0, 0>>>(dVals + dLoc, dFbrLikeSlcInds + dFbrIdxLoc, 
              dInds2 + dLoc, dFbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dFbrIdx1 + dFbrIdxLoc, TiledX[m].nFibers, 
              dU + dULoc[1], dU + dULoc[2], dU + dULoc[0], Opt.mode, Opt.R, warpPerFbr, logOfWarpPerFbr);

        else if (TiledX[0].ndims == 4) {
          mttkrp_MIHCSR_kernel_fbr_atomic_fbrLvlPar_4D<<<grid, block, 0, 0>>>(dVals + dLoc, dFbrLikeSlcInds + dFbrIdxLoc, 
              dInds3 + dLoc, dFbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dFbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, 
              TiledX[m].nFibers,  dU + dULoc[2], dU + dULoc[3], dU + dULoc[0], dU + dULoc[1], Opt.mode, Opt.R, warpPerFbr, logOfWarpPerFbr);
	}
      }

      else if(TiledX[m].modeOrder[TiledX[0].ndims-1] == MTTKRPmode && TiledX[m].totNnz){

        std::cout << "Run the implemention: nnz atomics\n";

        BLOCKSIZE = Opt.TBsize;
        dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

        int warpPerFbr = Opt.warpPerSlice; // default 4
        if(warpPerFbr > (BLOCKSIZE/32)){
          std::cout << "warpPerFbr (-w) cannot be higher than threadblock size! hint: increase -b!" << std::endl;
          return -1;
        }
        int logOfWarpPerFbr = log2(warpPerFbr);

        grid.x = ( warpPerFbr * 32 * TiledX[m].nFibers + BLOCKSIZE - 1) / BLOCKSIZE;

        if (TiledX[0].ndims == 3)
          mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar<<<grid, block, 0, 0>>>(dVals + dLoc, dFbrLikeSlcInds + dFbrIdxLoc, 
              dInds2 + dLoc, dFbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dFbrIdx1 + dFbrIdxLoc, TiledX[m].nFibers, 
              dU + dULoc[2], dU + dULoc[0], dU + dULoc[1], Opt.mode, Opt.R, warpPerFbr, logOfWarpPerFbr); 

        else if (TiledX[0].ndims == 4) {
          mttkrp_MIHCSR_kernel_all_atomic_fbrLvlPar_4D<<<grid, block, 0, 0>>>(dVals + dLoc, dFbrLikeSlcInds + dFbrIdxLoc, 
              dInds3 + dLoc, dFbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dFbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, 
              TiledX[m].nFibers,  dU + dULoc[3], dU + dULoc[0], dU + dULoc[1], dU + dULoc[2], Opt.mode, Opt.R, warpPerFbr, logOfWarpPerFbr);
	}
      }
      cudaDeviceSynchronize();
    }
  }

  /* Copying output matrix from GPU to CPU for correctness check */
  int MTTKRPmode = TiledX[0].ndims - 1;
  ITYPE loc = ((TiledX[0].ndims == 3) ? szDU[0] + szDU[1] : szDU[0] + szDU[1] + szDU[2]);
  checkCuda(cudaMemcpy(&U[MTTKRPmode].vals[0], dU + loc, 
        U[MTTKRPmode].nRows * U[MTTKRPmode].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), __LINE__);

  cudaFree(dVals); 
  cudaFree(dU);
  cudaFree(dFbrIdx0);
  cudaFree(dFbrIdx1);
  cudaFree(dFbrPtr0); 
  cudaFree(dfbrPtr1);
  cudaFree(dFbrLikeSlcInds);

  if(TiledX[0].ndims == 3)
    cudaFree(dInds2); 

  if(TiledX[0].ndims == 4){
    cudaFree(dFbrIdx2);
    cudaFree(dFbrPtr2);
    cudaFree(dInds3); 
  }
  delete[] dULoc;
  delete[] szDU;

  int totalMIslics = 0, totalMISfibers = 0, totalMIfibers = 0, totalMInnz = 0;;
  for (int m = 0; m <  TiledX[0].ndims; ++m){
    if(TiledX[m].totNnz){
      if(TiledX[m].ndims == 3){
        totalMIslics += TiledX[m].fbrIdx[0].size();
        totalMIfibers += TiledX[m].fbrPtr[1].size();
        totalMInnz += TiledX[m].totNnz;
      }

      if(TiledX[m].ndims == 4){
        totalMIslics += TiledX[m].fbrIdx[0].size();
        totalMISfibers += TiledX[m].fbrPtr[1].size();
        totalMIfibers += TiledX[m].fbrPtr[2].size();
        totalMInnz += TiledX[m].totNnz;
      }
    }
  }

  std::cout << "Resource usage: " << std::endl;
  if(TiledX[0].ndims == 3)
    std::cout << " nSlc:" << totalMIslics
              << ", nFibers:" << totalMIfibers << ", nnz:" << totalMInnz  
              << std::endl;
  else if(TiledX[0].ndims == 4)
    std::cout << " nSlc:" << totalMIslics  << ", nSFibers:" << totalMISfibers
              << ", nFibers:" << totalMIfibers << ", nnz:" << totalMInnz  
              << std::endl;
  return 0;
}
