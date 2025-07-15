#ifndef UTIL_H
#define UTIL_H

#define DTYPE float
#define ITYPE size_t // if chnage to unsigned int change the grid.x and gID in cuda kernel computation to long

#include <vector>
#include <algorithm>
#include <boost/sort/sort.hpp>
#include <iterator>
#include <unordered_map>
#include <map>
#include <boost/functional/hash.hpp>
#include <utility>  
#include <bits/stdc++.h>  
#include <time.h>
#include <sys/time.h>
#include <iomanip> 
#include <iostream>

using namespace std;

class Tensor{
  public:
    ITYPE ndims;
    ITYPE *dims;
    ITYPE totNnz;
    ITYPE nFibers;
    ITYPE *accessK;
    ITYPE *fbrLikeSlcInds;
    bool switchBC = false; // if true change matrix rand() to 1
    std::vector<ITYPE> modeOrder;
    std::vector<ITYPE> fbrCount;
    ITYPE **inds;
    DTYPE *vals;
    std::vector<vector<ITYPE>> fbrPtr;
    std::vector<vector<ITYPE>> fbrIdx;
    std::vector<vector<ITYPE>> slcMapperBin;
    ITYPE *nnzPerSlice;
    ITYPE *fiberPerSlice;
    ITYPE *nnzPerFiber;
    ITYPE *denseSlcPtr;
    ITYPE *partPerNnz;
    ITYPE *totnnzPerPart;
    unordered_map<pair<ITYPE, ITYPE>, ITYPE, boost::hash<pair<ITYPE, ITYPE>>> fbrHashTbl; 
};

class TiledTensor{
  public:
    ITYPE ndims;
    ITYPE *dims;
    ITYPE totNnz;
    ITYPE nFibers;
    ITYPE *accessK;
    ITYPE *fbrLikeSlcInds;
    std::vector<ITYPE> modeOrder;
    std::vector<ITYPE> fbrCount;
    ITYPE **inds;
    DTYPE *vals;
    std::vector<vector<ITYPE>> fbrPtr;
    std::vector<vector<ITYPE>> fbrIdx;
    std::vector<vector<ITYPE>> slcMapperBin;
    ITYPE *nnzPerSlice;
    ITYPE *fiberPerSlice;
    ITYPE *nnzPerFiber;
    ITYPE *denseSlcPtr;
    ITYPE *partPerNnz;
    ITYPE *totnnzPerPart;
    ITYPE *nnzInRank;
    ITYPE *fbrInRank;
    ITYPE *slcInRank;
    unordered_map<pair<ITYPE, ITYPE>, int, boost::hash<pair<ITYPE, ITYPE>>> fbrHashTbl; 
};

class HYBTensor{
  public:
    ITYPE ndims;
    ITYPE *dims;
    ITYPE totNnz;
    ITYPE HCSRnnz;
    ITYPE COOnnz;
    ITYPE CSLnnz;
    ITYPE nFibers;
    ITYPE *accessK;
    std::vector<ITYPE> modeOrder;
    ITYPE **inds;
    DTYPE *vals;
    std::vector<vector<ITYPE>> fbrPtr;
    std::vector<vector<ITYPE>> fbrIdx;
    std::vector<vector<ITYPE>> slcMapperBin;
    ITYPE **COOinds;
    DTYPE *COOvals;
    std::vector<ITYPE> CSLslicePtr;
    std::vector<ITYPE> CSLsliceIdx;
    ITYPE **CSLinds;
    DTYPE *CSLvals;
    std::vector<vector<ITYPE>> CSLslcMapperBin;

    HYBTensor(const Tensor &X) 
    { 
      ndims = X.ndims;
      dims = new ITYPE[X.ndims];
      totNnz = X.totNnz;
      for (int i = 0; i < ndims; ++i)
      {
        dims[i] = X.dims[i];
        modeOrder.push_back(X.modeOrder[i]);
      }
    } 
};

class Matrix{
  public:
    ITYPE nRows;
    ITYPE nCols;
    DTYPE *vals;
};

class semiSpTensor{
  public:
    ITYPE nRows;
    ITYPE nCols;
    DTYPE *vals;
};

class Options {
  public:
    ITYPE R = 32;
    ITYPE mode = 0;
    ITYPE warpPerSlice = 4;
    ITYPE nTile = 1;
    ITYPE tileSize;
    ITYPE gridSize = 512;
    ITYPE TBsize = 128;
    ITYPE MIfbTh = 1;
    ITYPE fiberPerWarp = 1;
    bool verbose = false;     // if true change matrix rand() to 1
    string inFileName; 
    string outFileName; 
    ITYPE nBin = 1;
    std::string m0 = "012";
    std::string m1 = "120";
    std::string m2 = "201";
    bool doCPD = false;
    ITYPE cpdIters = 10;
    bool natOrdering = false;
    int redunMode;
    ITYPE fbrThreashold = 99999999;

    void print() {
      std::cout << "R = " << R << '\n';
      std::cout << "mode = " << mode << '\n';
      std::cout << "warpPerSlice = " << warpPerSlice << '\n';
      std::cout << "nTiles = " << nTile << '\n';
      std::cout << "verbose = " << verbose << '\n';

      // must provide input file name 
      if(inFileName.empty()){
        std::cout << "Provide input file path. Program will exit." << std::endl;
        exit(0);
      }
      else{
        std::cout << "input file name = " << inFileName << '\n';
      }

      if(!outFileName.empty())
        std::cout << "output file name = " << outFileName << '\n';

    }
};

inline void check_opt(const Tensor &X, Options &Opt){

  if(X.ndims > 4){
    std::cout << "Supported tensor dimension is 3 or 4." << std::endl;
    exit(0);
  }

  if(Opt.mode > X.ndims - 1){
    std::cout << "Mode cannot be larger than tensor dimension." << std::endl;
    exit(0);
  }

  Opt.mode = 0;
} 

inline void order_tensormode(Tensor &X, const Options &Opt, const int mode){

  int *sortMode = new int[X.ndims]; //sorted according to mode length
  int *natMode = new int[X.ndims]; // natural ordering
  bool *taken = new bool[X.ndims];
  int *sortModeLen = new int[X.ndims];

  for (int m = 0; m < X.ndims; ++m){
    natMode[m] = (m + mode) % X.ndims;
    sortModeLen[m] = X.dims[natMode[m]];
    taken[m] = false;
  }

  if(Opt.natOrdering){
    for (int i = 0; i < X.ndims; ++i)
      X.modeOrder.push_back(natMode[i]);
    std::cout << "Natural mode ordering " << std::endl;
  }
  else{
    /*linear sort of dimension length*/   
    for (int i = 1; i < X.ndims; i++) {

      for (int j =i+1; j < X.ndims; j++) {

        if (sortModeLen[i] > sortModeLen[j]) 
          std::swap(sortModeLen[i],sortModeLen[j]);
      }
    }

    sortMode[0] = mode; 
    taken[mode] = true;

    for (int i = 1; i < X.ndims; i++) {

      for (int j = 0; j < X.ndims; j++) {

        if( sortModeLen[i] == X.dims[j] && !taken[j]){
          sortMode[i] = j;
          taken[j] = true;
          break;
        }
      }
    }

    for (int i = 0; i < X.ndims; ++i)    
      X.modeOrder.push_back(sortMode[i]);
  }

  if(Opt.verbose){
    std::cout << "mode ordering: ";
    for (int i = 0; i < X.ndims; ++i)
      std::cout << X.modeOrder[i] << " ";
    std::cout << std::endl;
  }

  delete[] sortMode;
  delete[] natMode;
  delete[] taken;
  delete[] sortModeLen;
}

inline int load_tensor(Tensor &X, const Options &Opt){

  if(Opt.verbose)
    std::cout << std::endl << "Loading tensor.." << std::endl;   

  string filename = Opt.inFileName;
  ITYPE index;
  DTYPE vid=0;

  ifstream fp(filename); 

  if(fp.fail()){
    std::cout << "File " << filename << " does not exist!" << std::endl;
    exit(0);
  }

  /*get number of line (totnnz)*/
  int numLines = 0;
  std::string unused;
  while ( std::getline(fp, unused) )
    ++numLines;
  X.totNnz = numLines - 2;

  fp.clear();                 // clear fail and eof bits
  fp.seekg(0, std::ios::beg);

  fp >> X.ndims; 

  X.dims = new ITYPE[X.ndims];

  for (int i = 0; i < X.ndims; ++i)
    fp >> X.dims[i]; 

  X.inds = new ITYPE*[X.ndims];  

  for(int i = 0; i < X.ndims; ++i)
    X.inds[i] = new ITYPE[X.totNnz];
  X.vals = new DTYPE[X.totNnz];   

  int idxCntr = 0;

  while(fp >> index) {

    X.inds[0][idxCntr] = index-1;
    for (int i = 1; i < X.ndims; ++i)
    {      
      fp >> index;
      X.inds[i][idxCntr] = index-1; 
    }
    fp >> vid;
    X.vals[idxCntr] = vid;
    idxCntr++;
  }

  order_tensormode(X, Opt, Opt.mode);

  return 0;
}

inline void init_tensor(Tensor *arrX, Tensor &X0, const Options &Opt, int mode){

  arrX[mode].ndims = X0.ndims;
  arrX[mode].dims = new ITYPE[arrX[mode].ndims];
  arrX[mode].totNnz = X0.totNnz;

  arrX[mode].inds = X0.inds;
  arrX[mode].vals = X0.vals;

  for (int i = 0; i < arrX[mode].ndims; ++i)
    arrX[mode].dims[i] = X0.dims[i];

  order_tensormode(arrX[mode], Opt, mode);
}

inline bool sort_pred_3D(tuple <ITYPE, ITYPE, ITYPE, DTYPE> left, 
    tuple <ITYPE, ITYPE, ITYPE, DTYPE> right) {

  if (get<0>(left) != get<0>(right)) 
    return (get<0>(left) < get<0>(right));

  return (get<1>(left) < get<1>(right));

}

inline bool sort_pred_4D(tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> left, 
    tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> right) {
  // return get<0>(left) < get<0>(right);

  if (get<0>(left) != get<0>(right)) 
    return (get<0>(left) < get<0>(right));

  if (get<1>(left) != get<1>(right)) 
    return (get<1>(left) < get<1>(right));

  return (get<2>(left) < get<2>(right));
}

inline void sort_COOtensor(Tensor &X){

  const ITYPE mode0 = X.modeOrder[0];
  const ITYPE mode1 = X.modeOrder[1];
  const ITYPE mode2 = X.modeOrder[2];
  ITYPE mode3;
  if(X.ndims == 4)
    mode3 = X.modeOrder[3];

  if(X.ndims == 3){

    vector < tuple <ITYPE, ITYPE, ITYPE, DTYPE> > items;
    tuple <ITYPE, ITYPE, ITYPE, DTYPE> ap;

    for (long idx = 0; idx < X.totNnz; ++idx) { 

      ap=std::make_tuple(X.inds[mode0][idx], X.inds[mode1][idx], X.inds[mode2][idx], X.vals[idx]);         
      items.push_back(ap);
    }
    // std::sort(std::parallel::par, items.begin(), items.end(), sort_pred);
    // std::sort(items.begin(), items.end(), sort_pred);
    boost::sort::sample_sort(items.begin(), items.end(), sort_pred_3D);

    for (long idx = 0; idx < X.totNnz; ++idx) {
      X.inds[mode0][idx] = get<0>(items[idx]);
      X.inds[mode1][idx] = get<1>(items[idx]);
      X.inds[mode2][idx] = get<2>(items[idx]);
      X.vals[idx] = get<3>(items[idx]);
    }
  }
  else if(X.ndims == 4){

    vector < tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> > items;
    tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> ap;

    for (long idx = 0; idx < X.totNnz; ++idx) { 

      ap=std::make_tuple(X.inds[mode0][idx], X.inds[mode1][idx], X.inds[mode2][idx], X.inds[mode3][idx], X.vals[idx]); 

      items.push_back(ap);
    }
    boost::sort::sample_sort(items.begin(), items.end(), sort_pred_4D);

    for (long idx = 0; idx < X.totNnz; ++idx) {

      X.inds[mode0][idx] = get<0>(items[idx]);
      X.inds[mode1][idx] = get<1>(items[idx]);
      X.inds[mode2][idx] = get<2>(items[idx]);           
      X.inds[mode3][idx] = get<3>(items[idx]);
      X.vals[idx] = get<4>(items[idx]);
    }
  }  
}

inline void sort_MI_CSF(const Tensor &X, TiledTensor *MTX, int m){

  const ITYPE mode0 = MTX[m].modeOrder[0];
  const ITYPE mode1 = MTX[m].modeOrder[1];
  const ITYPE mode2 = MTX[m].modeOrder[2];
  ITYPE mode3;
  if(X.ndims == 4)
    mode3 = MTX[m].modeOrder[3];

  if(X.ndims == 3){

    vector < tuple <ITYPE, ITYPE, ITYPE, DTYPE> > items;
    tuple <ITYPE, ITYPE, ITYPE, DTYPE> ap;

    for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 

      ap=std::make_tuple(MTX[m].inds[mode0][idx], MTX[m].inds[mode1][idx], MTX[m].inds[mode2][idx], MTX[m].vals[idx]); 
      items.push_back(ap);
    }

    // sort(items.begin(), items.end(), sort_pred);
    boost::sort::sample_sort(items.begin(), items.end(), sort_pred_3D);

    for (long idx = 0; idx < MTX[m].totNnz; ++idx) {

      MTX[m].inds[mode0][idx] = get<0>(items[idx]);
      MTX[m].inds[mode1][idx] = get<1>(items[idx]);
      MTX[m].inds[mode2][idx] = get<2>(items[idx]);
      MTX[m].vals[idx] = get<3>(items[idx]);
    }
  }

  else if(X.ndims == 4){

    vector < tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> > items;
    tuple <ITYPE, ITYPE, ITYPE, ITYPE, DTYPE> ap;

    for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 

      ap=std::make_tuple(MTX[m].inds[mode0][idx], MTX[m].inds[mode1][idx], MTX[m].inds[mode2][idx], MTX[m].inds[mode3][idx], MTX[m].vals[idx]); 
      items.push_back(ap);
    }
    boost::sort::sample_sort(items.begin(), items.end(), sort_pred_4D);

    for (long idx = 0; idx < MTX[m].totNnz; ++idx) {

      MTX[m].inds[mode0][idx] = get<0>(items[idx]);
      MTX[m].inds[mode1][idx] = get<1>(items[idx]);
      MTX[m].inds[mode2][idx] = get<2>(items[idx]);           
      MTX[m].inds[mode3][idx] = get<3>(items[idx]);
      MTX[m].vals[idx] = get<4>(items[idx]);
    }
  } 

  // std::cout << "sorted tile : " << m << std::endl;
  // for (long idx = 0; idx < MTX[m].totNnz; ++idx) { 
  // std::cout << MTX[m].inds[0][idx] << " "
  //           << MTX[m].inds[1][idx] << " "
  //           << MTX[m].inds[2][idx] << " "
  //           << MTX[m].vals[idx] <<  std::std::endl;
  // }
}

inline void print_HCSRtensor(const Tensor &X){

  std::cout << "no of fibers " << X.fbrPtr[1].size() << std::endl;

  ITYPE mode0 = X.modeOrder[0];
  ITYPE mode1 = X.modeOrder[1];
  ITYPE mode2 = X.modeOrder[2];

  for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

    for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){        

      for(ITYPE x = X.fbrPtr[1][fbr]; x < X.fbrPtr[1][fbr+1]; ++x) {
        if(mode0 == 0)
          std::cout << X.fbrIdx[0][slc] << " " << X.fbrIdx[1][fbr] << " " << X.inds[X.modeOrder[2]][x] << std::endl;
        if(mode0 == 1)
          std::cout  << X.inds[X.modeOrder[2]][x] << " "<< X.fbrIdx[0][slc] <<" "<<X.fbrIdx[1][fbr] << " " <<std::endl;
        if(mode0 == 2)
          std::cout  << X.fbrIdx[1][fbr]<<" "<< X.inds[X.modeOrder[2]][x]  << " "  << X.fbrIdx[0][slc] << std::endl;

      }            
    }
  }
}

inline void print_HCSRtensor_4D(const Tensor &X){

  std::cout << "no of fibers " << X.fbrPtr[1].size() << std::endl;

  ITYPE mode0 = X.modeOrder[0];
  ITYPE mode1 = X.modeOrder[1];
  ITYPE mode2 = X.modeOrder[2];
  ITYPE mode3 = X.modeOrder[3];

  for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

    for (int fbrS = X.fbrPtr[0][slc]; fbrS < X.fbrPtr[0][slc+1]; ++fbrS){   

      for (int fbr = X.fbrPtr[1][fbrS]; fbr < X.fbrPtr[1][fbrS+1]; ++fbr){        

        for(ITYPE x = X.fbrPtr[2][fbr]; x < X.fbrPtr[2][fbr+1]; ++x) {

          // if(mode0 == 0)
          std::cout << X.fbrIdx[0][slc] << " " << X.fbrIdx[1][fbrS] << " " << X.fbrIdx[2][fbr] << " " << X.inds[3][x] << std::endl;
          // if(mode0 == 1)
          //     std::cout  << X.fbrIdx[1][fbr] << " " << X.inds[1][x] << " "<< X.fbrIdx[0][slc]; << std::endl;
          // if(mode0 == 2)
          //     std::cout  << X.inds[0][x]  << " " << X.fbrIdx[0][slc]; << " " << X.fbrIdx[1][fbr]<< std::endl;

        }  
      }          
    }
  }
}

inline void print_TiledHCSRtensor(TiledTensor *TiledX, int tile){

  std::cout << "Tile " << tile << " of Tensor X in Tiled HCSR format: " << std::endl;

  const ITYPE mode0 = TiledX[tile].modeOrder[0];
  const ITYPE mode1 = TiledX[tile].modeOrder[1];
  const ITYPE mode2 = TiledX[tile].modeOrder[2];

  if(TiledX[tile].ndims == 3){
    for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {

      ITYPE idx0 = TiledX[tile].fbrIdx[0][slc]; //slc
      int fb_st = TiledX[tile].fbrPtr[0][slc];
      int fb_end = TiledX[tile].fbrPtr[0][slc+1];
      // printf("slc st- end: %d %d %d \n", slc, fb_st, fb_end );

      for (int fbr = fb_st; fbr < fb_end; ++fbr){        
        // printf("fbr %d :  ", fbr );    
        for(ITYPE x = TiledX[tile].fbrPtr[1][fbr]; x < TiledX[tile].fbrPtr[1][fbr+1]; ++x) {
          std::cout << idx0 << " " << TiledX[tile].inds[mode1][x] << " " << TiledX[tile].inds[mode2][x] << std::endl;

        }            
      }
    }
  }

  else if(TiledX[tile].ndims == 4){
    for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {

      for (int fbrS = TiledX[tile].fbrPtr[0][slc]; fbrS < TiledX[tile].fbrPtr[0][slc+1]; ++fbrS){   

        for (int fbr = TiledX[tile].fbrPtr[1][fbrS]; fbr < TiledX[tile].fbrPtr[1][fbrS+1]; ++fbr){        

          for(ITYPE x = TiledX[tile].fbrPtr[2][fbr]; x < TiledX[tile].fbrPtr[2][fbr+1]; ++x) {

            // if(mode0 == 0)
            std::cout << TiledX[tile].fbrLikeSlcInds[fbrS] << " " << TiledX[tile].fbrIdx[1][fbrS] << " " << TiledX[tile].fbrIdx[2][fbr] << " " << TiledX[tile].inds[3][x] << std::endl;
            // if(mode0 == 1)
            //     std::cout  << X.fbrIdx[1][fbr] << " " << X.inds[1][x] << " "<< X.fbrIdx[0][slc]; << std::endl;
            // if(mode0 == 2)
            //     std::cout  << X.inds[0][x]  << " " << X.fbrIdx[0][slc]; << " " << X.fbrIdx[1][fbr]<< std::endl;

          }  
        }          
      }
    }
  }
}

inline void create_HCSR(Tensor &X, const Options &Opt){

  ITYPE fbrThreashold = Opt.fbrThreashold;
  fbrThreashold = 99999999;//

  for (int i = 0; i < X.ndims - 1; ++i){
    X.fbrPtr.push_back(std::vector<ITYPE>());
    X.fbrIdx.push_back(std::vector<ITYPE>());
  }

  std::vector<ITYPE> prevId(X.ndims-1);
  std::vector<ITYPE> fbrId(X.ndims-1);

  for (int i = 0; i < X.ndims-1; ++i){
    prevId[i] =  X.inds[X.modeOrder[i]][0];
    X.fbrPtr[i].push_back(0);
    X.fbrIdx[i].push_back(prevId[i]);
    X.fbrPtr[i].reserve(X.totNnz);
    X.fbrIdx[i].reserve(X.totNnz);
  }

  int idx = 1 ;

  while(idx < X.totNnz) {

    for (int i = 0; i < X.ndims-1; ++i) 
      fbrId[i] = X.inds[X.modeOrder[i]][idx];

    ITYPE fiberNnz = 1;
    bool sameFbr = true;

    for (int i = 0; i < X.ndims-1; ++i) {
      if(fbrId[i] != prevId[i])
        sameFbr = false;
    }
    /* creating last fiber consisting all nonzeroes in same fiber */
    while( sameFbr && idx < X.totNnz && fiberNnz < fbrThreashold){
      ++idx;
      fiberNnz++;
      for (int i = 0; i < X.ndims-1; ++i) {
        fbrId[i] = X.inds[X.modeOrder[i]][idx];   
        if(fbrId[i] != prevId[i])
          sameFbr = false;
      }
    }

    if(idx == X.totNnz)
      break;

    /* X.ndims-2 is the last fiber ptr. Out of prev while loop means it is a new fiber. */
    X.fbrPtr[X.ndims-2].push_back(idx);
    X.fbrIdx[X.ndims-2].push_back(fbrId[X.ndims-2]);

    /* populating slice ptr and higher ptrs */
    for (int i = X.ndims - 3; i > -1 ; --i) {

      /* each dimension checks whether all parent/previous dimensions are in same slice/fiber/... */
      bool diffFbr = false;            
      int iDim = i;
      while(iDim > -1){
        if( fbrId[iDim] != prevId[iDim]) {//not else ..not become this in loop          
          diffFbr = true;
        } 
        iDim--;
      }
      if(diffFbr){
        X.fbrIdx[i].push_back(fbrId[i]);
        X.fbrPtr[i].push_back((ITYPE)(X.fbrPtr[i+1].size()) - 1);
      }
    }

    for (int i = 0; i < X.ndims-1; ++i)
      prevId[i] =  fbrId[i];

    ++idx;
    fiberNnz = 1;
  }
  X.fbrPtr[X.ndims-2].push_back(idx);
  X.fbrIdx[X.ndims-2].push_back(fbrId[X.ndims-2]);

  for (int i = X.ndims - 3; i > -1 ; --i)
    X.fbrPtr[i].push_back((ITYPE)(X.fbrPtr[i+1].size() - 1 ));

  X.nFibers = X.fbrPtr[1].size();

  // for (int i =0; i <  2 ;i++)
  //     X.inds[X.modeOrder[i]].resize(0);
}

inline void prefix_sum(ITYPE *x, ITYPE *y, int n){


  for (int i=0;i<=(log((double)n)-1);i++){

    for (int j=0;j<=n-1;j++)    {

      y[j] = x[j];

      if (j>=(powf(2,i))){
        int t=powf(2,i);
        y[j] += x[j-t];
      }
    }
    for (int j=0;j<=n-1;j++){
      x[j] = y[j];
    }
  }
}

inline void create_TiledHCSR(TiledTensor *TiledX, const Options &Opt, int tile){

  ITYPE fbrThreashold = Opt.fbrThreashold;
  ITYPE fbrSThreshold = 999999999;

  fbrSThreshold = 128;

  for (int i = 0; i < TiledX[tile].ndims - 1; ++i){
    TiledX[tile].fbrPtr.push_back(std::vector<ITYPE>());
    TiledX[tile].fbrIdx.push_back(std::vector<ITYPE>());
  }

  ITYPE mode0 = TiledX[tile].modeOrder[0];
  ITYPE mode1 = TiledX[tile].modeOrder[1];
  ITYPE mode2 = TiledX[tile].modeOrder[2];
  // ITYPE mode3 = TiledX[tile].modeOrder[3];

  std::vector<ITYPE> prevId(TiledX[tile].ndims-1);
  std::vector<ITYPE> fbrId(TiledX[tile].ndims-1);

  for (int i = 0; i < TiledX[tile].ndims-1; ++i){
    prevId[i] =  TiledX[tile].inds[TiledX[tile].modeOrder[i]][0];
    TiledX[tile].fbrPtr[i].push_back(0);
    TiledX[tile].fbrIdx[i].push_back(prevId[i]);
  }

  int idx = 1 ;
  ITYPE fiberSNnz = 1;

  while(idx < TiledX[tile].totNnz) {

    for (int i = 0; i < TiledX[tile].ndims-1; ++i) 
      fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];

    ITYPE fiberNnz = 1;
    bool sameFbr = true;

    for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
      if(fbrId[i] != prevId[i])
        sameFbr = false;
    }
    /* creating last fiber consisting all nonzeroes in same fiber */
    while( sameFbr && idx < TiledX[tile].totNnz && fiberNnz < fbrThreashold){
      ++idx;
      fiberNnz++;
      fiberSNnz++;
      for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
        fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];   
        if(fbrId[i] != prevId[i])
          sameFbr = false;
      }
    }

    if(idx == TiledX[tile].totNnz)
      break;

    /* TiledX[tile].ndims-2 is the last fiber ptr. Out of prev while loop means it is a new fiber. */
    TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
    TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);

    /* populating slice ptr and higher ptrs */
    for (int i = TiledX[tile].ndims - 3; i > -1 ; --i) {

      /* each dimension checks whether all parent/previous dimensions are in same slice/fiber/... */
      bool diffFbr = false;            
      int iDim = i;
      while(iDim > -1){
        if( fbrId[iDim] != prevId[iDim]) {//not else ..not become this in loop          
          diffFbr = true;
        } 
        /*splitting fbrS for 4D */
        else if( TiledX[tile].ndims == 4 && iDim == 1 && fiberSNnz > fbrSThreshold){ 
          diffFbr = true;                    
        }
        iDim--;
      }
      if(diffFbr){

        if(TiledX[tile].ndims == 4 && i == 1)
          fiberSNnz = 1;

        TiledX[tile].fbrIdx[i].push_back(fbrId[i]);
        TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size()) - 1);
      }
    }

    for (int i = 0; i < TiledX[tile].ndims-1; ++i)
      prevId[i] =  fbrId[i];

    ++idx;
    fiberSNnz++;
    fiberNnz = 1;

  }
  TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
  TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);

  for (int i = TiledX[tile].ndims - 3; i > -1 ; --i)
    TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size() - 1 ));

  TiledX[tile].nFibers = TiledX[tile].fbrPtr[1].size();

  // std::cout << tile << " nnz: " <<  TiledX[tile].totNnz << " nslices: " <<  TiledX[tile].fbrPtr[0].size()  << " nFibers: " << TiledX[tile].nFibers << std::endl;
  // if(tile == TiledX[tile].ndims - 1){
  //     if(TiledX[tile].totNnz){
  //         int totslc = TiledX[0].fbrPtr[0].size() + TiledX[1].fbrPtr[0].size() +TiledX[2].fbrPtr[0].size();
  //         int totFbr = TiledX[0].fbrPtr[1].size() + TiledX[1].fbrPtr[1].size() +TiledX[2].fbrPtr[1].size();
  //         std::cout << "Total slice: " << totslc << " " << totFbr << std::endl;
  //     }
  // }
}

inline void create_fbrLikeSlcInds(Tensor &X, const Options &Opt){

  X.fbrLikeSlcInds = (ITYPE *)malloc( X.nFibers * sizeof(ITYPE));

  for(ITYPE slc = 0; slc < X.fbrIdx[0].size(); ++slc) {

    for (int fbr = X.fbrPtr[0][slc]; fbr < X.fbrPtr[0][slc+1]; ++fbr){  

      X.fbrLikeSlcInds[fbr] =   X.fbrIdx[0][slc] ;     
    }
  }
}

inline void create_fbrLikeSlcInds(TiledTensor *TiledX, int mode){

  TiledX[mode].fbrLikeSlcInds = (ITYPE *)malloc( TiledX[mode].nFibers * sizeof(ITYPE));

  for(ITYPE slc = 0; slc < TiledX[mode].fbrIdx[0].size(); ++slc) {

    for (int fbr = TiledX[mode].fbrPtr[0][slc]; fbr < TiledX[mode].fbrPtr[0][slc+1]; ++fbr){  

      TiledX[mode].fbrLikeSlcInds[fbr] =   TiledX[mode].fbrIdx[0][slc] ;     
    }
  }
}


inline int populate_paritions(Tensor &X, TiledTensor *MTX){

  // avoid pushback by using tot nnzperpart
  ITYPE *nnzCntr = new ITYPE[X.ndims];
  memset(nnzCntr, 0, X.ndims * sizeof(ITYPE));  

  int mode;
  for (int idx = 0; idx < X.totNnz; ++idx){
    mode = X.partPerNnz[idx];
    X.totnnzPerPart[mode]++;
  }

  for(int i = 0; i < X.ndims; ++i){

    MTX[i].inds = new ITYPE*[X.ndims];
    MTX[i].totNnz = X.totnnzPerPart[i];

    for(int m = 0; m < X.ndims; ++m){

      MTX[i].inds[m] = new ITYPE[X.totnnzPerPart[i]];
    }
    MTX[i].vals = new DTYPE[X.totnnzPerPart[i]];
  }

  for (int idx = 0; idx < X.totNnz; ++idx){

    int mode = X.partPerNnz[idx];

    for (int i = 0; i < X.ndims; ++i)  {
      MTX[mode].inds[i][nnzCntr[mode]] = X.inds[i][idx]; 
    }
    MTX[mode].vals[nnzCntr[mode]] = X.vals[idx];    
    nnzCntr[mode]++;
  }

  delete[] nnzCntr;
  return 0;
}

inline int binarySearch1(ITYPE *arr, ITYPE left, ITYPE right, ITYPE value) { 

  while (left <= right) {
    // int middle = (left + right) / 2;
    int middle = ((unsigned int)left+(unsigned int)right) >> 1;
    if (arr[middle] == value)
      return middle;
    else if (arr[middle] > value)
      right = middle - 1;
    else
      left = middle + 1;
  }
  return -1;
}

inline int binarySearch(ITYPE *arr, ITYPE l, ITYPE r, ITYPE x) { 

  if (r >= l) { 
    // int mid = ((unsigned int)left+(unsigned int)right) >> 1;
    // ITYPE mid = l + (r - l) / 2; 
    unsigned int mid = ((unsigned int)l + (unsigned int)r) >> 1;

    if (arr[mid] == x) 
      return mid; 

    if (arr[mid] > x) 
      return binarySearch(arr, l, mid - 1, x); 

    return binarySearch(arr, mid + 1, r, x); 
  } 
  return -1; 
}

inline int maxOf3( int a, int b, int c )
{
  int max = ( a < b ) ? b : a;
  return ( ( max < c ) ? c : max );
}

inline void mm_partition_reuseBased(Tensor *arrX, Tensor &X, TiledTensor *MTX, Options & Opt){

  X.partPerNnz = new ITYPE[X.totNnz];
  memset(X.partPerNnz, 0, X.totNnz * sizeof(ITYPE));  
  X.totnnzPerPart = new ITYPE[X.ndims];
  memset(X.totnnzPerPart, 0, X.ndims * sizeof(ITYPE));  

  for (int m = 0; m < arrX[0].ndims; ++m){

    if(m != Opt.redunMode){

      int sliceMode=arrX[m].modeOrder[0];
      int fiberMode=arrX[m].modeOrder[1];

      arrX[m].nnzPerFiber = new ITYPE[arrX[m].nFibers];
      memset(arrX[m].nnzPerFiber, 0, arrX[m].nFibers * sizeof(ITYPE));     

      arrX[m].nnzPerSlice = new ITYPE[arrX[m].dims[sliceMode]];
      memset(arrX[m].nnzPerSlice, 0, arrX[m].dims[sliceMode] * sizeof(ITYPE));  

      arrX[m].denseSlcPtr = (ITYPE*)malloc( (arrX[m].dims[sliceMode]+1) * sizeof(ITYPE)); //new ITYPE[arrX[m].dims[sliceMode]];
      memset(arrX[m].denseSlcPtr, 0, (arrX[m].dims[sliceMode] + 1 ) * sizeof(ITYPE));  
    }
  }

  /*creating dense slices so that nnz can directly index slices unlike fiber. For
    fiber it needs to scan all fibers in a slice. */

  for (int m = 0; m < arrX[0].ndims; ++m){

    // if(m == Opt.redunMode) continue;
    if(m != Opt.redunMode){

      {
        for(ITYPE slc = 0; slc < arrX[m].fbrIdx[0].size(); ++slc) {

          arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]] = arrX[m].fbrPtr[0][slc];

          if(slc == arrX[m].fbrIdx[0].size()-1)
            arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc];
          else
            arrX[m].denseSlcPtr[arrX[m].fbrIdx[0][slc]+1] = arrX[m].fbrPtr[0][slc+1];

          /* Populate nnz per fiber and nnz per slice */
          for (int fbr = arrX[m].fbrPtr[0][slc]; fbr < arrX[m].fbrPtr[0][slc+1]; ++fbr){      

            if(X.ndims == 3){   
              arrX[m].nnzPerFiber[fbr] = arrX[m].fbrPtr[1][fbr+1] - arrX[m].fbrPtr[1][fbr];
              arrX[m].nnzPerSlice[arrX[m].fbrIdx[0][slc]] += arrX[m].nnzPerFiber[fbr];
            }

            else if(X.ndims == 4){  

              for (int fbrIn = arrX[m].fbrPtr[1][fbr]; fbrIn < arrX[m].fbrPtr[1][fbr+1]; ++fbrIn)          
                arrX[m].nnzPerFiber[fbr] += arrX[m].fbrPtr[2][fbrIn+1] - arrX[m].fbrPtr[2][fbrIn];

              arrX[m].nnzPerSlice[arrX[m].fbrIdx[0][slc]] += arrX[m].nnzPerFiber[fbr];
            }
          }
        }
      }
    }
  }

  //int threshold = ( X.totNnz / X.dims[0] + X.totNnz / X.dims[1] + X.totNnz / X.dims[2]) / 3;
  //int thNnzInTile = X.totNnz*1;

  /* initialize MICSF tiles */
  int mode = 0;

  for (int m = 0; m < X.ndims; ++m){

    MTX[m].ndims = X.ndims;
    MTX[m].dims = new ITYPE[MTX[m].ndims];  
    MTX[m].totNnz = 0; // WHY mode?

    for (int i = 0; i < X.ndims; ++i){
      MTX[m].modeOrder.push_back(arrX[m].modeOrder[i]); 
      MTX[m].dims[i] = X.dims[i];
    }     
  }    

  /* Populate with nnz for each slice for each mode */

  //ITYPE mode0 = 0;//std::min(X.dims[0], X.dims[1], X.dims[2]);
  //ITYPE mode1 = 1;//X.modeOrder[1];
  //ITYPE mode2 = 2;//X.modeOrder[2];
  //ITYPE mode3 = 3;

  //not mode sorted
  int shortestMode = ( (X.dims[X.modeOrder[0]] <= X.dims[X.modeOrder[1]]) ? X.modeOrder[0] : X.modeOrder[1]) ;

  bool sameFm0m1 = false, sameFm0m2 = false, sameFm1m2 = false, sameFm0m3 = false, 
       sameFm1m3 = false, sameFm2m3 = false;

  int fbTh =  Opt.MIfbTh;
  //int slTh =  1, 
  int shortMode = 0;
  int longMode = -1;

  for (int m = 0; m < X.ndims; ++m){

    if(m == 1){
      if (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
        sameFm0m1 = true;
        shortMode = (arrX[m].dims[0] <= arrX[m].dims[1] ? 0 : 1);
        longMode = (arrX[m].dims[0] <= arrX[m].dims[1] ? 1 : 0);
      }
    }
    else if(m == 2){
      if(arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]){
        sameFm0m2 = true;
        shortMode = (arrX[m].dims[0] <= arrX[m].dims[2] ? 0 : 2);
        longMode = (arrX[m].dims[0] <= arrX[m].dims[2] ? 2 : 0);
      }
      else if ( arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
        sameFm1m2 = true;
        shortMode = (arrX[m].dims[1] <= arrX[m].dims[2] ? 1 : 2);
        longMode = (arrX[m].dims[1] <= arrX[m].dims[2] ? 2 : 1);
      }
    }
    else if(m == 3){
      if(arrX[m].modeOrder[1] == arrX[m-3].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-3].modeOrder[1]){
        sameFm0m3 = true;
        shortMode = (arrX[m].dims[0] <= arrX[m].dims[3] ? 0 : 3);
        longMode = (arrX[m].dims[0] <= arrX[m].dims[3] ? 3 : 0);
      }
      else if ( arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]){
        sameFm1m3 = true;
        shortMode = (arrX[m].dims[1] <= arrX[m].dims[3] ? 1 : 3);
        longMode = (arrX[m].dims[1] <= arrX[m].dims[3] ? 3 : 1);
      }
      else if ( arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
        sameFm2m3 = true;
        shortMode = (arrX[m].dims[2] <= arrX[m].dims[3] ? 2 : 3);
        longMode = (arrX[m].dims[2] <= arrX[m].dims[3] ? 3 : 2);
      }
    }
  }

  bool casePr = false;

  /******** Process NNZ********s*/
  ITYPE *fbrNnz = new ITYPE[X.ndims];
  ITYPE *fbrNo = new ITYPE[X.ndims];
  ITYPE *curIdx = new ITYPE[X.ndims];
  ITYPE *sliceNnz = new ITYPE[X.ndims];
  ITYPE tmpSlc;
  int nonSelMode ;

  for (int idx = 0; idx < X.totNnz; ++idx){

    bool modeDone = false;

    for (int m = 0; m < X.ndims; ++m)
      curIdx[m] = X.inds[m][idx];

    /*Finding fiber nnz*/
    for (int m = 0; m < X.ndims; ++m){
      int nextMode = arrX[m].modeOrder[1];

      //change to sameFm*m*
      if((m == 1 && sameFm0m1) || (m == 2 && sameFm1m2) || (m == 3 && sameFm2m3)){
        fbrNnz[m] = fbrNnz[m - 1];
        fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
      }
      else if((m == 2 && sameFm0m2) || (m == 3 && sameFm1m3)){
        fbrNnz[m] = fbrNnz[m - 2];
        fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
      }
      else if(m == 3 && sameFm0m3){
        fbrNnz[m] = fbrNnz[m - 3];
        fbrNo[m] = 99999;//curIdx[arrX[m].modeOrder[1]];
      }
      else{
        ITYPE result, tmp;
        ITYPE idx_j = curIdx[arrX[m].modeOrder[1]];
        tmpSlc = curIdx[m];

        /*binary search*/
        {
          int n =  arrX[m].denseSlcPtr[tmpSlc+1] - arrX[m].denseSlcPtr[tmpSlc];//sizeof(arr) / sizeof(arr[0]); 
          ITYPE fbr = arrX[m].denseSlcPtr[tmpSlc];  
          result = binarySearch1(&(arrX[m].fbrIdx[1][fbr]), 0, n, idx_j); 
          tmp = arrX[m].nnzPerFiber[result+fbr];
          fbrNo[m] = result+fbr;
          fbrNnz[m] = tmp;
        }
      }
    }

    if(X.ndims == 3){

      //changing to > = from >

      if ( fbrNnz[0] >=  fbTh * std::max(fbrNnz[1] , fbrNnz[2]) && !modeDone) {
        modeDone = true;
        if(sameFm0m1 || sameFm0m2 || sameFm0m3){
          mode = shortMode;
        }
        else{
          mode = 0;
        }
      }
      else if ( fbrNnz[1] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[2]) && !modeDone) {
        modeDone = true;
        if(sameFm1m2 || sameFm1m3)
          mode = shortMode;
        else 
          mode = 1;
      }
      else if ( fbrNnz[2] >=  fbTh * std::max(fbrNnz[0] , fbrNnz[1]) && !modeDone) {
        modeDone = true;
        if(sameFm2m3)
          mode = shortMode;
        else 
          mode = 2;
      }
    }

    else if(X.ndims == 4){

      if ( fbrNnz[0] >=  fbTh * maxOf3(fbrNnz[1] , fbrNnz[2], fbrNnz[3]) && !modeDone) {
        modeDone = true;
        if(sameFm0m1 || sameFm0m2 || sameFm0m3)
          mode = shortMode;
        else
          mode = 0;
      }
      else if ( fbrNnz[1] >=  fbTh * maxOf3(fbrNnz[0] , fbrNnz[2], fbrNnz[3]) && !modeDone) {
        modeDone = true;
        if(sameFm1m2 || sameFm1m3)
          mode = shortMode;
        else 
          mode = 1;
      }
      else if ( fbrNnz[2] >=  fbTh * maxOf3(fbrNnz[0] , fbrNnz[1], fbrNnz[3]) && !modeDone) {
        modeDone = true;
        if(sameFm2m3)
          mode = shortMode;
        else 
          mode = 2;
      }
      else if ( fbrNnz[3] >=  fbTh * maxOf3(fbrNnz[0] , fbrNnz[1], fbrNnz[2]) && !modeDone) {
        modeDone = true;
        mode = 3;
      }
    }

    // slcNnzPerParti[mode][curIdx[mode]]++;

    if(!modeDone)
      mode = shortestMode;//mode = -1;

    //fr_m 
    // if(mode == 1)
    //     nonSelMode = 0;
    // else nonSelMode = 1; 
    //nell-2
    for (int i = 0; i < X.ndims; ++i)  {
      if(mode == shortMode){
        if( i != shortMode && i != longMode){
          nonSelMode = i;
          arrX[nonSelMode].nnzPerFiber[fbrNo[nonSelMode]]--;
        }
      }
      else{
        if(i != Opt.redunMode && i != mode){
          nonSelMode = i;
          arrX[nonSelMode].nnzPerFiber[fbrNo[nonSelMode]]--;
        }  
      }
    }
    // if(mode == 1)
    //     nonSelMode = 2;
    // else if(mode == 2) 
    //     nonSelMode = 0; 

    // arrX[nonSelMode].nnzPerFiber[fbrNo[nonSelMode]]--;
    mode = 2;
    /*populate new partitions*/
    if(mode > -1){
      X.partPerNnz[idx] = mode;
    }
    if(casePr) 
      std::cout << "selected mode: " << mode << std::endl;
  }

  delete[] fbrNnz;
  delete[] fbrNo;
  delete[] curIdx;
  delete[] sliceNnz;
}
// more detailed check

inline void create_mats(const Tensor &X, Matrix *U, const Options &Opt, bool ata){

  ITYPE mode;
  ITYPE R = Opt.R;
  for (int m = 0; m < X.ndims; ++m){  
    mode = X.modeOrder[m];
    U[mode].nRows =  X.dims[mode];
    U[mode].nCols =  R;
    if(ata)  
      U[mode].nRows = U[mode].nCols;
    U[mode].vals = (DTYPE*)malloc(U[mode].nRows * U[mode].nCols * sizeof(DTYPE));
  }
}

// jli added
inline DTYPE RandomValue(void)
{
  DTYPE v =  3.0 * ((DTYPE) rand() / (DTYPE) RAND_MAX);
  if(rand() % 2 == 0) {
    v *= -1;
  }
  return v;
}


inline void randomize_mats(const Tensor &X, Matrix *U, const Options &Opt){

  ITYPE mode;

  for (int m = 0; m < X.ndims; ++m){  
    mode = X.modeOrder[m];
    srand48(123L);
    for(long r = 0; r < U[mode].nRows; ++r){
      for(long c = 0; c < U[mode].nCols; ++c){ // or u[mode].nCols 

        if(Opt.doCPD)
          U[mode].vals[r * U[mode].nCols + c] = RandomValue(); //mode + .5;//1.5 * (mode+1);;// .1 * drand48(); //1 ;//; //
        else
          U[mode].vals[r * U[mode].nCols + c] = mode + .5;//1.5
      }
    }
  }
}

inline void zero_mat(const Tensor &X, Matrix *U, ITYPE mode){

  for(long r = 0; r < U[mode].nRows; ++r){
    for(long c = 0; c < U[mode].nCols; ++c) // or u[mode].nCols 
      U[mode].vals[r * U[mode].nCols +c] = 0;
  }
}

inline void print_matrix(Matrix *U, ITYPE mode){

  std::cout << U[mode].nRows << " x " << U[mode].nCols << " matrix" << std::endl;
  std::cout << std::fixed;
  for (int i = 0; i < 3; ++i)
    // for (int i = U[mode].nRows-5; i <  U[mode].nRows; ++i)
  {
    // for (int j = 0; j < U[mode].nCols; ++j)
    for (int j = 0; j < 3; ++j)
    {
      std::cout << std::setprecision(2) << U[mode].vals[i * U[mode].nCols + j] << "\t" ;
    }
    std::cout << std::endl;  
  }
}

inline void write_output(Matrix *U, ITYPE mode, string outFile){

  ofstream fp(outFile); 
  fp << U[mode].nRows << " x " << U[mode].nCols << " matrix" << std::endl;
  fp << std::fixed;
  for (int i = 0; i < U[mode].nRows; ++i)
  {
    for (int j = 0; j < U[mode].nCols; ++j)
    {
      fp << std::setprecision(2) << U[mode].vals[i * U[mode].nCols + j] << "\t" ;
    }
    fp << std::endl;  
  }
}

inline void correctness_check(DTYPE *out, DTYPE *COOout, int nr, int nc){
  long mismatch = 0;
  DTYPE maxDiff = 0;
  DTYPE precision = 0.1;
  std::cout << std::fixed;
  for (int i = 0; i < nr; ++i){
    for (int j = 0; j < nc; ++j){
      DTYPE diff = abs(out[i * nc + j] - COOout[i * nc + j]);
      if( diff > precision){
        if(diff > maxDiff)
          maxDiff = diff;
        if(mismatch < 5 && j == 0)
          std::cout << "mismatch at (" << i <<"," << j <<") got: " << out[i * nc +j] << " exp: " << COOout[i * nc +j] << ". "<< std::endl;
        mismatch++;
      }          
    }
  }

  if(mismatch == 0)
    std::cout << "PASS!" << std::endl;
  else{
    std::cout <<  mismatch <<" mismatches found at " << precision << " precision. " << std::endl;
    std::cout << "Maximum diff " << maxDiff << ". "<<std::endl;
  }
}

inline double seconds(){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

inline void print_help_and_exit() {
  printf("options:\n\
      -R rank/feature : set the rank (default 32)\n\
      -m mode : set the mode of MTTKRP (default 0)\n\
      -w warp per slice: set number of WARPs assign to per slice  (default 4)\n\
      -i input file name: e.g., ../dataset/delicious.tns \n\
      -o output file name: optional dump\n");

  exit(1);
}

inline Options parse_cmd_options(int argc, char **argv) {

  Options param;
  int i;
  //handle options
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-')
      break;
    if (++i >= argc){
      print_help_and_exit();

    }

    switch (argv[i - 1][1]) {
      case 'R':
        param.R = atoi(argv[i]);
        break;
      case 'm':
        param.mode = atoi(argv[i]);
        break;

      case 'w':
        param.warpPerSlice = atoi(argv[i]);
        break;

      case 'l':
        param.nTile = atoi(argv[i]);
        break;

      case 'f':
        param.fbrThreashold = atoi(argv[i]);
        break;

      case 'b':
        param.TBsize = atoi(argv[i]);
        break;

      case 's':
        param.fiberPerWarp = atoi(argv[i]);
        break;

      case 'h':
        param.MIfbTh = atoi(argv[i]);
        break;

      case 'g':
        param.gridSize = atoi(argv[i]);
        break;

      case 'v':
        if(atoi(argv[i]) == 1)
          param.verbose = true;
        else
          param.verbose = false;
        break;

      case 'i':
        param.inFileName = argv[i];
        break;

      case 'o':
        param.outFileName = argv[i];
        break;

      case 'p':
        param.m0 = argv[i];
        break;

      case 'q':
        param.m1 = argv[i];
        break;

      case 'r':
        param.m2 = argv[i];
        break;

      default:
        fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
        print_help_and_exit();
        break;
    }
  }


  if (i > argc){
    std::cout << "weird " << argc << std::endl;
    print_help_and_exit();
  }

  return param;
}

#endif


