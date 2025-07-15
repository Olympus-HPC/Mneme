/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * Copyright (c) 2013, Istvan Reguly and others. 
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * The name of Mike Giles may not be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Istvan Reguly ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @brief initial version of mutli-material code with full dense matrix representaiton
 * @author Istvan Reguly
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#ifdef KNL
#include <hbwmalloc.h>
#else
#define hbw_malloc malloc
#define hbw_free free
#endif

struct full_data
{
  int sizex;
  int sizey;
  int Nmats;
  double * __restrict__ rho;
  double * __restrict__ rho_mat_ave;
  double * __restrict__ p;
  double * __restrict__ Vf;
  double * __restrict__ t;
  double * __restrict__ V;
  double * __restrict__ x;
  double * __restrict__ y;
  double * __restrict__ n;
  double * __restrict__ rho_ave;
};

struct compact_data
{
  int sizex;
  int sizey;
  int Nmats;
  double * __restrict__ rho_compact;
  double * __restrict__ rho_compact_list;
  double * __restrict__ rho_mat_ave_compact;
  double * __restrict__ rho_mat_ave_compact_list;
  double * __restrict__ p_compact;
  double * __restrict__ p_compact_list;
  double * __restrict__ Vf_compact_list;
  double * __restrict__ t_compact;
  double * __restrict__ t_compact_list;
  double * __restrict__ V;
  double * __restrict__ x;
  double * __restrict__ y;
  double * __restrict__ n;
  double * __restrict__ rho_ave_compact;
  int * __restrict__ imaterial;
  int * __restrict__ matids;
  int * __restrict__ nextfrac;
  int * __restrict__ mmc_index;
  int * __restrict__ mmc_i;
  int * __restrict__ mmc_j;
  int mm_len;
  int mmc_cells;
};


extern void full_matrix_cell_centric(full_data cc);

extern void full_matrix_material_centric(full_data cc, full_data mc);

extern bool full_matrix_check_results(full_data cc, full_data mc);

extern void compact_cell_centric(full_data cc, compact_data ccc, int argc, char** argv);

extern bool compact_check_results(full_data cc, compact_data ccc);

void initialise_field_rand(full_data cc, double prob2, double prob3, double prob4) {

  //let's use a morton space filling curve here
  srand(0);
  double prob1 = 1.0-prob2-prob3-prob4;
#ifdef DEBUG
  printf("Random layout %g %g %g %g\n", prob1, prob2, prob3, prob4);
#endif

  for (int n = 0; n < cc.sizex*cc.sizey; n++) {
    int i = n%cc.sizex;//n & 0xAAAA;
    int j = n/cc.sizex;//n & 0x5555;

    double r = (double)rand()/(double)RAND_MAX;
    int m = (double)rand()/(double)RAND_MAX * cc.Nmats/4 + (cc.Nmats/4)*(n/(cc.sizex*cc.sizey/4));
    int m2, m3, m4;
    cc.rho[(i+cc.sizex*j)*cc.Nmats+m] = 1.0;
    cc.t[(i+cc.sizex*j)*cc.Nmats+m] = 1.0;
    cc.p[(i+cc.sizex*j)*cc.Nmats+m] = 1.0;
    if (r >= prob1) {
      m2 = (double)rand()/(double)RAND_MAX * cc.Nmats/4 + (cc.Nmats/4)*(n/(cc.sizex*cc.sizey/4));
      while (m2 == m)
        m2 = (double)rand()/(double)RAND_MAX * cc.Nmats/4 + (cc.Nmats/4)*(n/(cc.sizex*cc.sizey/4));
      cc.rho[(i+cc.sizex*j)*cc.Nmats+m2] = 1.0;
      cc.t[(i+cc.sizex*j)*cc.Nmats+m2] = 1.0;
      cc.p[(i+cc.sizex*j)*cc.Nmats+m2] = 1.0;
    }
    if (r >= 1.0-prob4-prob3) {
      m3 = (double)rand()/(double)RAND_MAX * cc.Nmats/4 + (cc.Nmats/4)*(n/(cc.sizex*cc.sizey/4));
      while (m3 == m && m3 == m2)
        m3 = (double)rand()/(double)RAND_MAX * cc.Nmats/4 + (cc.Nmats/4)*(n/(cc.sizex*cc.sizey/4));
      cc.rho[(i+cc.sizex*j)*cc.Nmats+m3] = 1.0;
      cc.t[(i+cc.sizex*j)*cc.Nmats+m3] = 1.0;
      cc.p[(i+cc.sizex*j)*cc.Nmats+m3] = 1.0;
    }
    if (r >= 1.0-prob4) {
      m4 = (double)rand()/(double)RAND_MAX * cc.Nmats/4 + (cc.Nmats/4)*(n/(cc.sizex*cc.sizey/4));
      while (m4 == m && m4 == m2 && m4 == m3)
        m4 = (double)rand()/(double)RAND_MAX * cc.Nmats/4 + (cc.Nmats/4)*(n/(cc.sizex*cc.sizey/4));
      cc.rho[(i+cc.sizex*j)*cc.Nmats+m4] = 1.0;
      cc.t[(i+cc.sizex*j)*cc.Nmats+m4] = 1.0;
      cc.p[(i+cc.sizex*j)*cc.Nmats+m4] = 1.0;
    }
  }
}

void initialise_field_static(full_data cc) {
  //Pure cells and simple overlaps
  int sizex = cc.sizex;
  int sizey = cc.sizey;
  int Nmats = cc.Nmats;
  int width = sizex/Nmats;

  int overlap_i = std::max(0.0,ceil((double)sizey/1000.0)-1);
  int overlap_j = std::max(0.0,floor((double)sizex/1000.0)-1);

  //Top
  for (int mat = 0; mat < cc.Nmats/2; mat++) {
#pragma omp parallel for
    for (int j = mat*width; j < sizey/2+overlap_j; j++) {
      for (int i = mat*width-(mat>0)-(mat>0)*overlap_i; i < (mat+1)*width; i++) { //+1 for overlap
        cc.rho[(i+sizex*j)*cc.Nmats+mat] = 1.0;
        cc.t[(i+sizex*j)*cc.Nmats+mat] = 1.0;
        cc.p[(i+sizex*j)*cc.Nmats+mat] = 1.0;
      }
      for (int i = sizex-mat*width-1+(mat>0)*overlap_i; i >= sizex-(mat+1)*width-1; i--) { //+1 for overlap
        cc.rho[(i+sizex*j)*cc.Nmats+mat] = 1.0;
        cc.t[(i+sizex*j)*cc.Nmats+mat] = 1.0;
        cc.p[(i+sizex*j)*cc.Nmats+mat] = 1.0;
      }
    }

#pragma omp parallel for
    for (int j = mat*width-(mat>0)-(mat>0)*overlap_j; j < (mat+1)*width; j++) { //+1 for overlap
      for (int i = mat*width-(mat>0)-(mat>0)*overlap_i; i < sizex-mat*width; i++) {
        cc.rho[(i+sizex*j)*cc.Nmats+mat] = 1.0;
        cc.t[(i+sizex*j)*cc.Nmats+mat] = 1.0;
        cc.p[(i+sizex*j)*cc.Nmats+mat] = 1.0;
      }
    }
  }

  //Bottom
  for (int mat = 0; mat < cc.Nmats/2; mat++) {
#pragma omp parallel for
    for (int j = sizey/2-1-overlap_j; j < sizey-mat*width; j++) {
      for (int i = mat*width-(mat>0)-(mat>0)*overlap_i; i < (mat+1)*width; i++) { //+1 for overlap
        cc.rho[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
        cc.t[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
        cc.p[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
      }
      for (int i = sizex-mat*width-1+(mat>0)*overlap_i; i >= sizex-(mat+1)*width-1; i--) { //+1 for overlap
        cc.rho[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
        cc.t[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
        cc.p[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
      }
    }
#pragma omp parallel for
    for (int j = sizey-mat*width-1+(mat>0)*overlap_j; j >= sizey-(mat+1)*width-(mat<(cc.Nmats/2-1)); j--) { //+1 for overlap
      for (int i = mat*width; i < sizex-mat*width; i++) {
        cc.rho[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
        cc.t[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
        cc.p[(i+sizex*j)*cc.Nmats+mat+cc.Nmats/2] = 1.0;
      }
    }
  }
  //Fill in corners
#pragma omp parallel for
  for (int mat = 1; mat < cc.Nmats/2; mat++) {
    for (int j = sizey/2-3; j < sizey/2-1;j++)
      for (int i = 2; i < 5+overlap_i; i++) {
        //x neighbour material
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;
        //y neighbour material
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;
        //x-y neighbour material
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;
      }
    for (int j = sizey/2; j < sizey/2+2+overlap_j;j++)
      for (int i = 2; i < 5; i++) {
        //x neighbour material
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats+cc.Nmats/2+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats+cc.Nmats/2+mat] = 1.0;
        //y neighbour material
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;

      }
  }
  int only_8 = 0;
  for (int mat = cc.Nmats/2+1; mat < cc.Nmats; mat++) {
    for (int j = sizey/2-3; j < sizey/2-1;j++)
      for (int i = 2; i < 5; i++) {
        //x neighbour material
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;
        //y neighbour material
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;
      }
    for (int j = sizey/2; j < sizey/2+2;j++)
      for (int i = 2; i < 4; i++) {
        if (i < 3 && only_8<6) {
          //y neighbour material
          cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;
          cc.rho[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;
        }
        if (i==2 && only_8==0) {
          //x-y neighbour material
          cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;
          cc.rho[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 1.0;
        }
        //x neighbour material
        if (mat >= cc.Nmats-8 && j==sizey/2+1 && i==3) if (only_8++>=4) {
          break;
        }
        cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats+mat-1] = 1.0;
        cc.rho[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.t[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;cc.p[(mat*width-i+sizex*j)*cc.Nmats+mat] = 1.0;
      }
  }
#pragma omp parallel for
  for (int mat=cc.Nmats/2+1; mat < cc.Nmats/2+5; mat++) {
    int i = 2; int j = sizey/2+1;
    cc.rho[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat] = 0.0;cc.t[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 0.0;cc.p[(mat*width+i-2+sizex*j)*cc.Nmats-cc.Nmats/2+mat-1] = 0.0;
  }
}

void initialise_field_file(full_data cc) {
  int sizex = cc.sizex;
  int sizey = cc.sizey;
  int Nmats = cc.Nmats;

  int status;
  FILE *fp;
  fp = fopen("volfrac.dat", "r");
  if (!fp) {
    fprintf(stderr, "unable to read volume fractions from file \"%s\"\n",
        "volfrac.dat");
    exit(-1);
  }

  int nmats;
  status = fscanf(fp, "%d", &nmats);
  if (status < 0) {
    printf("error in read at line %d\n",__LINE__);
    exit(1);
  }
  if (nmats != Nmats) {
    printf("Error, invalid Nmats: %d!=%d\n", nmats, Nmats);
    exit(1);
  }
  if (sizex%1000 != 0 || sizey%1000!=0) {
    printf("size needs to be an integer multiple of 1000x1000: %dx%d\n", sizex, sizey);
    exit(1);
  }
  int sx = sizex/1000;
  int sy = sizey/1000;

  status = fscanf(fp, "%d", &nmats);
  if (status < 0) {
    printf("error in read at line %d\n",__LINE__);
    exit(1);
  }

  for (int j = 0; j < sizey; j++)
    for (int i = 0; i < sizex; i++)
      for (int m = 0; m < nmats; m++)
        cc.Vf[(i+sizex*j)*Nmats+m] = 0.0;

  char matname[256];
  for (int m = 0; m < nmats; m++){
    status = fscanf(fp, "%s", matname);            // read and discard
    if (status < 0) {
      printf("error in read at line %d\n",__LINE__);
      exit(1);
    }
  }

  for (int j = 0; j < 1000; j++)
    for (int i = 0; i < 1000; i++)
      for (int m = 0; m < nmats; m++) {
        double volfrac;
        status = fscanf(fp, "%lf", &(volfrac));
        if (status < 0) {
          printf("error in read at line %d\n",__LINE__);
          exit(1);
        }
        if (volfrac > 0.0) {
          for (int jj = 0; jj < sy; jj++)
            for (int ii = 0; ii < sx; ii++) {
              cc.Vf[(i*sx+ii+sizex*(j*sy+jj))*Nmats+m] = volfrac;
              cc.rho[(i*sx+ii+sizex*(j*sy+jj))*Nmats+m] = 1.0;
              cc.t[(i*sx+ii+sizex*(j*sy+jj))*Nmats+m] = 1.0;
              cc.p[(i*sx+ii+sizex*(j*sy+jj))*Nmats+m] = 1.0;
            }
        }
      }
  fclose(fp);

}

int main(int argc, char** argv) {
  int sizex = 1000;
  if (argc > 1)
    sizex = atoi(argv[1]);
  int sizey = 1000;
  if (argc > 2)
    sizey = atoi(argv[2]);
  int ncells = sizex*sizey;

  int Nmats = 50;

  full_data cc;
  full_data mc;
  compact_data ccc;

  cc.sizex = sizex;
  mc.sizex = sizex;
  ccc.sizex = sizex;
  cc.sizey = sizey;
  mc.sizey = sizey;
  ccc.sizey = sizey;
  cc.Nmats = Nmats;
  mc.Nmats = Nmats;
  ccc.Nmats = Nmats;

  //Allocate the four state variables for all Nmats materials and all cells 
  //density
  cc.rho =  (double*)malloc(Nmats*ncells*sizeof(double));
  memset(cc.rho, 0, Nmats*ncells*sizeof(double));
  //average density in neighbourhood
  cc.rho_mat_ave =  (double*)malloc(Nmats*ncells*sizeof(double));
  memset(cc.rho_mat_ave, 0, Nmats*ncells*sizeof(double));
  //pressure
  cc.p = (double*)malloc(Nmats*ncells*sizeof(double));
  memset(cc.p, 0, Nmats*ncells*sizeof(double));
  //Fractional volume
  cc.Vf = (double*)malloc(Nmats*ncells*sizeof(double));
  memset(cc.Vf, 0, Nmats*ncells*sizeof(double));
  //temperature
  cc.t = (double*)malloc(Nmats*ncells*sizeof(double));
  memset(cc.t, 0, Nmats*ncells*sizeof(double));

  // Buffers for material-centric representation
  //density
  mc.rho =  (double*)malloc(Nmats*ncells*sizeof(double));
  //average density in neighbouring cells
  mc.rho_mat_ave =  (double*)malloc(Nmats*ncells*sizeof(double));
  memset(mc.rho_mat_ave, 0, Nmats*ncells*sizeof(double));

  //pressure
  mc.p = (double*)malloc(Nmats*ncells*sizeof(double));
  //Fractional volume
  mc.Vf = (double*)malloc(Nmats*ncells*sizeof(double));
  //temperature
  mc.t = (double*)malloc(Nmats*ncells*sizeof(double));

  //Allocate per-cell only datasets
  cc.V = (double*)malloc(ncells*sizeof(double));
  cc.x = (double*)malloc(ncells*sizeof(double));
  cc.y = (double*)malloc(ncells*sizeof(double));

  //Allocate per-material only datasets
  cc.n = (double*)malloc(Nmats*sizeof(double)); // number of moles

  //Allocate output datasets
  cc.rho_ave = (double*)malloc(ncells*sizeof(double));
  mc.rho_ave = (double*)malloc(ncells*sizeof(double));
  ccc.rho_ave_compact = (double*)hbw_malloc(ncells*sizeof(double));

  // Cell-centric compact storage
  ccc.rho_compact = (double*)hbw_malloc(ncells*sizeof(double));
  ccc.rho_mat_ave_compact = (double*)hbw_malloc(ncells*sizeof(double));
  memset(ccc.rho_mat_ave_compact, 0, ncells*sizeof(double));
  ccc.p_compact = (double*)hbw_malloc(ncells*sizeof(double));
  ccc.t_compact = (double*)hbw_malloc(ncells*sizeof(double));

  int *nmats = (int*)hbw_malloc(ncells*sizeof(int));
  ccc.imaterial = (int*)hbw_malloc(ncells*sizeof(int));

  // List
  double mul = ceil((double)sizex/1000.0) * ceil((double)sizey/1000.0);
  int list_size = mul * 49000 * 2 + 600 * 3 + 400 * 4;
  if (argc>=6)
    list_size = (double(sizex*sizey)*atof(argv[3])*2+double(sizex*sizey)*atof(argv[4])*3+double(sizex*sizey)*atof(argv[5])*4)*1.1;


  //plain linked list
  ccc.nextfrac = (int*)hbw_malloc(list_size*sizeof(int));
  int *frac2cell = (int*)hbw_malloc(list_size*sizeof(int));
  ccc.matids = (int*)hbw_malloc(list_size*sizeof(int));

  //CSR list
  ccc.mmc_index = (int *)hbw_malloc(list_size*sizeof(int)); //CSR mapping for mix cell idx -> compact list position
  ccc.mmc_i = (int *)hbw_malloc(list_size*sizeof(int)); // mixed cell -> physical cell i coord
  ccc.mmc_j = (int *)hbw_malloc(list_size*sizeof(int)); //  mixed cell -> physical cell j coord



  ccc.mmc_cells = 0;
  ccc.Vf_compact_list = (double*)hbw_malloc(list_size*sizeof(double));
  ccc.rho_compact_list = (double*)hbw_malloc(list_size*sizeof(double));
  ccc.rho_mat_ave_compact_list = (double*)hbw_malloc(list_size*sizeof(double));
  memset(ccc.rho_mat_ave_compact_list, 0, list_size*sizeof(double));
  ccc.t_compact_list = (double*)hbw_malloc(list_size*sizeof(double));
  ccc.p_compact_list = (double*)hbw_malloc(list_size*sizeof(double));

  int imaterial_multi_cell;

  //Initialise arrays
  double dx = 1.0/sizex;
  double dy = 1.0/sizey;
  for (int j = 0; j < sizey; j++) {
    for (int i = 0; i < sizex; i++) {
      cc.V[i+j*sizex] = dx*dy;
      cc.x[i+j*sizex] = dx*i;
      cc.y[i+j*sizex] = dy*j;
    }
  }

  for (int mat = 0; mat < Nmats; mat++) {
    cc.n[mat] = 1.0; // dummy value
  }

  //These are the same throughout
  ccc.V = mc.V = cc.V;
  ccc.x = mc.x = cc.x;
  ccc.y = mc.y = cc.y;
  ccc.n = mc.n = cc.n;

  if (argc>=6) initialise_field_rand(cc, atof(argv[3]), atof(argv[4]), atof(argv[5]));
  else initialise_field_file(cc);
  //else initialise_field_static(cc);

  FILE *f = nullptr;
  int print_to_file = 0;

  if (print_to_file==1)
    FILE *f = fopen("map.txt","w");

  //Compute fractions and count cells
  int cell_counts_by_mat[4] = {0,0,0,0};
  ccc.mmc_cells = 0;
  for (int j = 0; j < sizey; j++) {
    for (int i = 0; i < sizex; i++) {
      int count = 0;
      for (int mat = 0; mat < Nmats; mat++) {
        count += cc.rho[(i+sizex*j)*Nmats+mat]!=0.0;
      }
      if (count == 0) {
        printf("Error: no materials in cell %d %d\n",i,j);
        int mat = 1;
        cc.rho[(i+sizex*j)*Nmats+mat] = 1.0;cc.t[(i+sizex*j)*Nmats+mat] = 1.0;cc.p[(i+sizex*j)*Nmats+mat] = 1.0; cc.Vf[(i+sizex*j)*Nmats+mat] = 1.0;
        mc.rho[ncells*mat + i+sizex*j] = 1.0;mc.t[ncells*mat + i+sizex*j] = 1.0;mc.p[ncells*mat + i+sizex*j] = 1.0; mc.Vf[ncells*mat + i+sizex*j] = 1.0;
        count = 1;
      }
      if (count > 1) ccc.mmc_cells++;

      cell_counts_by_mat[count-1]++;

      if (print_to_file==1) {
        if (i!=0) fprintf(f,", %d",count);
        else fprintf(f,"%d",count);
      }

      if (argc>=6) //Only if rand - file read has Volfrac already
        for (int mat = 0; mat < Nmats; mat++) {
          if (cc.rho[(i+sizex*j)*Nmats+mat]!=0.0) cc.Vf[(i+sizex*j)*Nmats+mat]=1.0/count;
        }
    }
    if (print_to_file==1)
      fprintf(f,"\n");
  }
#ifdef DEBUG
  printf("Pure cells %d, 2-materials %d, 3 materials %d, 4 materials %d: MMC cells %d\n",
      cell_counts_by_mat[0],cell_counts_by_mat[1],cell_counts_by_mat[2],cell_counts_by_mat[3], ccc.mmc_cells);
#endif

  if (cell_counts_by_mat[1]*2+cell_counts_by_mat[2]*3+cell_counts_by_mat[3]*4 >= list_size) {
    printf("ERROR: list_size too small\n");
    exit(-1);
  }
  if (print_to_file==1)
    fclose(f);

  // Convert representation to material-centric (using extra buffers)
#pragma omp parallel for
  for (int j = 0; j < sizey; j++) {
    for (int i = 0; i < sizex; i++) {
      for (int mat = 0; mat < Nmats; mat++) {
        mc.rho[ncells*mat + i+sizex*j] = cc.rho[(i+sizex*j)*Nmats+mat];
        mc.p[ncells*mat + i+sizex*j] = cc.p[(i+sizex*j)*Nmats+mat];
        mc.Vf[ncells*mat + i+sizex*j] = cc.Vf[(i+sizex*j)*Nmats+mat];
        mc.t[ncells*mat + i+sizex*j] = cc.t[(i+sizex*j)*Nmats+mat];
      }
    }
  }

  // Copy data from cell-centric full matrix storage to cell-centric compact storage
  imaterial_multi_cell = 0;
  ccc.mmc_cells = 0;
  for (int j = 0; j < sizey; j++) {
    for (int i = 0; i < sizex; i++) {
      int mat_indices[4] = { -1, -1, -1, -1 };
      int matindex = 0;
      int count = 0;

      for (int mat = 0; mat < Nmats; mat++) {
        if (cc.rho[(i+sizex*j)*Nmats+mat]!=0.0) {
          mat_indices[matindex++] = mat;
          count += 1;
        }
      }

      if (count == 0) {
        printf("Error: no materials in cell %d %d\n",i,j);
        int mat = 1;
        cc.rho[(i+sizex*j)*Nmats+mat] = 1.0;cc.t[(i+sizex*j)*Nmats+mat] = 1.0;cc.p[(i+sizex*j)*Nmats+mat] = 1.0; cc.Vf[(i+sizex*j)*Nmats+mat] = 1.0;
        mc.rho[ncells*mat + i+sizex*j] = 1.0;mc.t[ncells*mat + i+sizex*j] = 1.0;mc.p[ncells*mat + i+sizex*j] = 1.0; mc.Vf[ncells*mat + i+sizex*j] = 1.0;
        count = 1;
      }

      if (count == 1) {
        int mat = mat_indices[0];
        ccc.rho_compact[i+sizex*j] = cc.rho[(i+sizex*j)*Nmats+mat];
        ccc.p_compact[i+sizex*j] = cc.p[(i+sizex*j)*Nmats+mat];
        ccc.t_compact[i+sizex*j] = cc.t[(i+sizex*j)*Nmats+mat];
        nmats[i+sizex*j] = -1;
        // NOTE: HACK: we index materials from zero, but zero can be a list index
        ccc.imaterial[i+sizex*j] = mat + 1;
      }
      else { // count > 1
        nmats[i+sizex*j] = count;
        // note the minus sign, it needs to be negative
#ifdef LINKED
        ccc.imaterial[i+sizex*j] = -imaterial_multi_cell;
#else
        ccc.imaterial[i+sizex*j] = -ccc.mmc_cells;
#endif
        ccc.mmc_index[ccc.mmc_cells] = imaterial_multi_cell;
        ccc.mmc_i[ccc.mmc_cells] = i;
        ccc.mmc_j[ccc.mmc_cells] = j;
        ccc.mmc_cells++;

        for (int list_idx = imaterial_multi_cell; list_idx < imaterial_multi_cell + count; ++list_idx) {
          // if last iteration
          if (list_idx == imaterial_multi_cell + count - 1)
            ccc.nextfrac[list_idx] = -1;
          else // not last
            ccc.nextfrac[list_idx] = list_idx + 1;

          frac2cell[list_idx] = i+sizex*j;

          int mat = mat_indices[list_idx - imaterial_multi_cell];
          ccc.matids[list_idx] = mat;

          ccc.Vf_compact_list[list_idx] = cc.Vf[(i+sizex*j)*Nmats+mat];
          ccc.rho_compact_list[list_idx] = cc.rho[(i+sizex*j)*Nmats+mat];
          ccc.p_compact_list[list_idx] = cc.p[(i+sizex*j)*Nmats+mat];
          ccc.t_compact_list[list_idx] = cc.t[(i+sizex*j)*Nmats+mat];
        }

        imaterial_multi_cell += count;
      }
    }
  }
  ccc.mmc_index[ccc.mmc_cells] = imaterial_multi_cell;
  ccc.mm_len = imaterial_multi_cell;

  full_matrix_cell_centric(cc);
  /*  full_matrix_material_centric(cc, mc);
  // Check results
  if (!full_matrix_check_results(cc, mc)) {
  goto end;
  }*/
#define MIN(a,b) (a)<(b)?(a):(b)
  double a1,a2,a3;

  compact_cell_centric(cc, ccc, argc, argv);

  int cell_mat_count = 1*cell_counts_by_mat[0] + 2*cell_counts_by_mat[1]
    + 3*cell_counts_by_mat[2] + 4*cell_counts_by_mat[3];
  //Alg 1:
  size_t alg1 = 0;
  //read imaterial (sizex*sizey)*sizeof(int)
  alg1 += (sizex*sizey)*sizeof(int);
  //read Vf (cell_mat_count - cell_counts_by_mat[0])*sizeof(double)
  alg1 += (cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
#ifdef FUSED
  //write rho_ave_compact (sizex*sizey)*sizeof(double)
  alg1 += (sizex*sizey)*sizeof(double);
  //read V (sizex*sizey)*sizeof(double)
  alg1 += (sizex*sizey)*sizeof(double);
  //read rho_compact+list cell_mat_count*sizeof(double)
  alg1 += cell_mat_count*sizeof(double);
  //LINKED - read nextfrac (cell_mat_count - cell_counts_by_mat[0])*sizeof(int)
#ifdef LINKED
  alg1 += (cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //CSR - read mmc_index (ccc.mmc_cells+1) * sizeof(int)
#else
  alg1 += (ccc.mmc_cells+1) * sizeof(int);
#endif
#else
  //write rho_ave_compact (sizex*sizey+ccc.mmc_cells)*sizeof(double)
  alg1 += (sizex*sizey+ccc.mmc_cells)*sizeof(double);
  //read V (sizex*sizey+ccc.mmc_cells)*sizeof(double)
  alg1 += (sizex*sizey+ccc.mmc_cells)*sizeof(double);
  //read rho_compact+list (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double)
  alg1 += (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //CSR - read mmc_index (ccc.mmc_cells+1) * sizeof(int)
  alg1 += (ccc.mmc_cells+1) * sizeof(int);
  //CSR - read mmc_i&j (ccc.mmc_cells) * 2 * sizeof(int)
  alg1 += (ccc.mmc_cells) * 2 * sizeof(int);
#endif

  //Alg2
  size_t alg2 = 0;
  //read imaterial (sizex*sizey)*sizeof(int)
  alg2 += (sizex*sizey)*sizeof(int);
  //read Vf (cell_mat_count - cell_counts_by_mat[0])*sizeof(double)
  alg2 += (cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //read matids (cell_mat_count - cell_counts_by_mat[0])*sizeof(int)
  alg2 += (cell_mat_count - cell_counts_by_mat[0])*sizeof(int);
#ifdef FUSED
  //read rho_compact+list cell_mat_count*sizeof(double)
  alg2 += cell_mat_count*sizeof(double);
  //read t_compact+list cell_mat_count*sizeof(double)
  alg2 += cell_mat_count*sizeof(double);
  //read p_compact+list cell_mat_count*sizeof(double)
  alg2 += cell_mat_count*sizeof(double);
  //read n Nmats*sizeof(double)
  alg2 += Nmats*sizeof(double);
  //LINKED - read nextfrac (cell_mat_count - cell_counts_by_mat[0])*sizeof(int)
#ifdef LINKED
  alg2 += (cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //CSR - read mmc_index (ccc.mmc_cells+1) * sizeof(int)
#else
  alg2 += (ccc.mmc_cells+1) * sizeof(int);
#endif

#else //FUSED
  //read rho_compact+list (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double)
  alg2 += (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //read t_compact+list (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double)
  alg2 += (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //read p_compact+list (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double)
  alg2 += (sizex*sizey+cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //CSR - read mmc_index (ccc.mmc_cells+1) * sizeof(int)
  alg2 += (ccc.mmc_cells+1) * sizeof(int);
  //read n Nmats*sizeof(double)
  alg2 += Nmats*sizeof(double);
#endif


  //Alg3
  size_t alg3 = 0;
  //read x & y
  alg3 += 2*sizex*sizey*sizeof(double);
  //read imaterial (sizex*sizey)*sizeof(int)
  alg3 += (sizex*sizey)*sizeof(int);
  //write rho_mat_ave_compact+list cell_mat_count*sizeof(double)
  alg3 += cell_mat_count*sizeof(double);
  //read matids (cell_mat_count - cell_counts_by_mat[0])*sizeof(int)
  alg3 += (cell_mat_count - cell_counts_by_mat[0])*sizeof(int);
  //read rho_compact+list cell_mat_count*sizeof(double)
  alg3 += cell_mat_count*sizeof(double);
  //LINKED - read nextfrac (cell_mat_count - cell_counts_by_mat[0])*sizeof(int)
#ifdef LINKED
  alg3 += (cell_mat_count - cell_counts_by_mat[0])*sizeof(double);
  //CSR - read mmc_index (ccc.mmc_cells+1) * sizeof(int)
#else
  alg3 += (ccc.mmc_cells+1) * sizeof(int);
#endif


  // Check results
  if (!compact_check_results(cc, ccc))
  {
    goto end;
  }

end:
  free(mc.rho); free(mc.p); free(mc.Vf); free(mc.t);
  free(cc.rho_mat_ave); free(mc.rho_mat_ave); hbw_free(ccc.rho_mat_ave_compact); hbw_free(ccc.rho_mat_ave_compact_list);
  free(cc.rho); free(cc.p); free(cc.Vf); free(cc.t);
  free(cc.V); free(cc.x); free(cc.y);
  free(cc.n);
  free(cc.rho_ave); free(mc.rho_ave); hbw_free(ccc.rho_ave_compact);

  hbw_free(ccc.rho_compact); hbw_free(ccc.p_compact); hbw_free(ccc.t_compact);
  hbw_free(nmats); hbw_free(ccc.imaterial);
  hbw_free(ccc.nextfrac); hbw_free(frac2cell); hbw_free(ccc.matids);
  hbw_free(ccc.Vf_compact_list); hbw_free(ccc.rho_compact_list);
  hbw_free(ccc.t_compact_list); hbw_free(ccc.p_compact_list);
  return 0;
}
