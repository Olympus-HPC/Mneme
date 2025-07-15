/*****************************************************************************
 This file is part of the XLQC program.                                      
 Copyright (C) 2015 Xin Li <lixin.reco@gmail.com>                            
                                                                           
 Filename:  cuda_rys_dp.h                                                      
 License:   BSD 3-Clause License

 * The implementation of Rys quadrature routines in C is taken from the
 * PyQuante quantum chemistry program, Copyright (c) 2004, Richard P. Muller.
 * PyQuante version 1.2 and later is covered by the modified BSD license. 
 * Please see int_lib/LICENSE.
 
 This software is provided by the copyright holders and contributors "as is"
 and any express or implied warranties, including, but not limited to, the
 implied warranties of merchantability and fitness for a particular purpose are
 disclaimed. In no event shall the copyright holder or contributors be liable
 for any direct, indirect, incidental, special, exemplary, or consequential
 damages (including, but not limited to, procurement of substitute goods or
 services; loss of use, data, or profits; or business interruption) however
 caused and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of the use
 of this software, even if advised of the possibility of such damage.
 *****************************************************************************/

#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAXROOTS 7

__device__ void cuda_Roots_dp(int n, double X, double roots[], double weights[]);
__device__ void cuda_Root123_dp(int n, double X, double roots[], double weights[]);
__device__ void cuda_Root4_dp(double X, double roots[], double weights[]);
__device__ void cuda_Root5_dp(double X, double roots[], double weights[]);
__device__ void cuda_Root6_dp(int n,double X, double roots[], double weights[]);
__device__ double cuda_Int1d_dp(int i, int j,int k, int l,
                                double xi, double xj, double xk, double xl,
                                double alpha_ij_A, double alpha_kl_B, double sqrt_AB,
                                double A, double B, double Px, double Qx,
                                double inv_t1, double B00, double B1, double B1p, 
                                double G[][MAXROOTS]);
__device__ double cuda_rys_coulomb_repulsion_dp(double xa,double ya,double za,double norma,
                                                int la,int ma,int na,double alphaa,
                                                double xb,double yb,double zb,double normb,
                                                int lb,int mb,int nb,double alphab,
                                                double xc,double yc,double zc,double normc,
                                                int lc,int mc,int nc,double alphac,
                                                double xd,double yd,double zd,double normd,
                                                int ld,int md,int nd,double alphad);

__device__ double cuda_rys_pbf_dp(const double *ptr_i, const double *ptr_j, 
                                  const double *ptr_k, const double *ptr_l);

__global__ void cuda_mat_J_PI_dp(const double *pbf_xlec, const int *pbf_to_cbf, int n_pbf,
                                 const double *mat_D, double *mat_J_PI, const double *mat_Q);

__global__ void cuda_mat_K_PI_dp(const double *pbf_xlec, const int *pbf_to_cbf, int n_pbf,
                                 const double *mat_D, double *mat_K_PI, const double *mat_Q);
