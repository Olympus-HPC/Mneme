#include <fstream>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <cmath> 
#include <bits/stdc++.h> 
#include "mttkrp_cpu.h"
#include "mttkrp_gpu.h" 


int main(int argc, char* argv[]){ 

  Options Opt = parse_cmd_options(argc, argv);

  printf("\nStarting MTTKRP..\n");

  Tensor X;
  load_tensor(X, Opt);
  double t0;

  t0 = seconds();
  sort_COOtensor(X);
  printf("Sort time : %.3f sec \n", seconds() - t0); 

  check_opt(X, Opt); //check options are good

  Matrix *U = new Matrix[X.ndims]; 
  create_mats(X, U, Opt, false);
  randomize_mats(X, U, Opt);

  // Collect slice and fiber stats: Create CSF for all modes
  int redunMode = -1;

  Tensor *arrX = new Tensor[X.ndims]; 


  for (int m = 0; m < X.ndims; ++m){

    init_tensor(arrX, X, Opt, m);

    if(m == 1){
      if (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && 
          arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
        redunMode = m;
      }
    }
    else if(m == 2){
      if ((arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]) || 
          (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])){
        redunMode = m;
      }
    }

    else if(m == 3){
      if ((arrX[m].modeOrder[1] == arrX[m-3].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-3].modeOrder[1]) ||
          (arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]) ||
          (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])){
        redunMode = m;
      }
    }

    if(m !=  redunMode) sort_COOtensor(arrX[m]);

    if(m !=  redunMode) create_HCSR(arrX[m], Opt); 
  }       

  Opt.redunMode = redunMode;

  TiledTensor ModeWiseTiledX[X.ndims];
  t0 = seconds();
  mm_partition_reuseBased(arrX, X, ModeWiseTiledX, Opt);
  populate_paritions(X, ModeWiseTiledX);
  printf("mm_partition & populate - time: %.3f sec \n", seconds() - t0);

  t0 = seconds();
  for (int m = 0; m < X.ndims; ++m){

    if(ModeWiseTiledX[m].totNnz > 0){           
      sort_MI_CSF(X, ModeWiseTiledX, m);
      create_TiledHCSR(ModeWiseTiledX, Opt, m);
      create_fbrLikeSlcInds(ModeWiseTiledX, m);
    }
  }
  printf("Sort,createCSF,createFbrIND - time: %.3f sec \n", seconds() - t0);

  printf("Starting MM-CSF on a GPU\n");

  t0 = seconds();
  MTTKRP_MIHCSR_GPU(ModeWiseTiledX, U, Opt);
  printf("Total GPU time: %.3f sec \n", seconds() - t0);

  if(!Opt.outFileName.empty()){
    printf("Save results to a file\n");
    write_output(U, Opt.mode, Opt.outFileName);
  }

  Opt.mode = X.ndims-1;
  int mode = Opt.mode;
  int nr = U[mode].nRows;  
  int nc = U[mode].nCols;
  DTYPE *out = (DTYPE*)malloc(nr * nc * sizeof(DTYPE));
  memcpy(out, U[mode].vals, nr*nc * sizeof(DTYPE));

  print_matrix(U, mode);
  randomize_mats(X, U, Opt);
  zero_mat(X, U, mode);

  printf("Correctness check with COO on mode %d.\n", mode);
  ((X.ndims == 3) ?  MTTKRP_COO_CPU(X, U, Opt) :  MTTKRP_COO_CPU_4D(X, U, Opt));

  print_matrix(U, mode);
  correctness_check(out, U[mode].vals, nr, nc);

  free(out);
  for (int m = 0; m < X.ndims; ++m){  
    mode = X.modeOrder[m];
    free(U[mode].vals);
  }
  for(int i = 0; i < X.ndims; ++i){
    for(int m = 0; m < X.ndims; ++m)
      delete[] ModeWiseTiledX[i].inds[m];
    delete[] ModeWiseTiledX[i].vals;
    delete[] ModeWiseTiledX[i].inds;
    delete[] X.inds[i]; 
  }
  delete[] X.inds;
  delete[] X.vals;
  delete[] X.dims;
  delete[] X.partPerNnz;
  delete[] X.totnnzPerPart;

  delete[] U; 
  for(int m = 0; m < X.ndims; ++m) {
    delete[] arrX[m].dims; 
    if(m != Opt.redunMode){
      delete[] arrX[m].nnzPerFiber; 
      delete[] arrX[m].nnzPerSlice; 
      free(arrX[m].denseSlcPtr);
    }
  }
  delete[] arrX; 
  return 0;
}
