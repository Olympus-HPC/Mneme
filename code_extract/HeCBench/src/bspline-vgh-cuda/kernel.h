const int WSIZE = 12000;          // Walker
const int NSIZE = 2003;           // Values
const int MSIZE = NSIZE*3+3;      // Gradient vectors
const int OSIZE = NSIZE*9+9;      // Hessian Matrices 

__device__
static inline void eval_UBspline_3d_s_vgh (
    const float *__restrict__ coefs_init,
    const intptr_t xs,
    const intptr_t ys,
    const intptr_t zs,
    float *__restrict__ vals,
    float *__restrict__ grads,
    float *__restrict__ hess,
    const float *a,
    const float *b,
    const float *c,
    const float *da,
    const float *db,
    const float *dc,
    const float *d2a,
    const float *d2b,
    const float *d2c,
    const float dxInv,
    const float dyInv,
    const float dzInv)
{
  float h[9];
  float v0 = 0.f;
  for (int i = 0; i < 9; ++i) h[i] = 0.f;

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++){
      float pre20 = d2a[i]*  b[j];
      float pre10 =  da[i]*  b[j];
      float pre00 =   a[i]*  b[j];
      float pre11 =  da[i]* db[j];
      float pre01 =   a[i]* db[j];
      float pre02 =   a[i]*d2b[j];

      const float *coefs = coefs_init + i*xs + j*ys;

      float sum0 =   c[0] * coefs[0] +   c[1] * coefs[zs] +   c[2] * coefs[zs*2] +   c[3] * coefs[zs*3];
      float sum1 =  dc[0] * coefs[0] +  dc[1] * coefs[zs] +  dc[2] * coefs[zs*2] +  dc[3] * coefs[zs*3];
      float sum2 = d2c[0] * coefs[0] + d2c[1] * coefs[zs] + d2c[2] * coefs[zs*2] + d2c[3] * coefs[zs*3];

      h[0]  += pre20 * sum0;
      h[1]  += pre11 * sum0;
      h[2]  += pre10 * sum1;
      h[4]  += pre02 * sum0;
      h[5]  += pre01 * sum1;
      h[8]  += pre00 * sum2;
      h[3]  += pre10 * sum0;
      h[6]  += pre01 * sum0;
      h[7]  += pre00 * sum1;
      v0    += pre00 * sum0;
    }
  vals[0] = v0;
  grads[0]  = h[3] * dxInv;
  grads[1]  = h[6] * dyInv;
  grads[2]  = h[7] * dzInv;

  hess [0] = h[0]*dxInv*dxInv;
  hess [1] = h[1]*dxInv*dyInv;
  hess [2] = h[2]*dxInv*dzInv;
  hess [3] = h[1]*dxInv*dyInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [4] = h[4]*dyInv*dyInv;
  hess [5] = h[5]*dyInv*dzInv;
  hess [6] = h[2]*dxInv*dzInv; // Copy hessian elements into lower half of 3x3 matrix
  hess [7] = h[5]*dyInv*dzInv; //Copy hessian elements into lower half of 3x3 matrix
  hess [8] = h[8]*dzInv*dzInv;
}

__global__ void bspline (
    const float *__restrict__ spline_coefs,
    const intptr_t xs,
    const intptr_t ys,
    const intptr_t zs,
          float*__restrict__ walkers_vals, 
          float*__restrict__ walkers_grads,
          float*__restrict__ walkers_hess,
    const float*__restrict__ a,
    const float*__restrict__ b,
    const float*__restrict__ c,
    const float*__restrict__ da,
    const float*__restrict__ db,
    const float*__restrict__ dc,
    const float*__restrict__ d2a,
    const float*__restrict__ d2b,
    const float*__restrict__ d2c,
    const float spline_x_grid_delta_inv, 
    const float spline_y_grid_delta_inv, 
    const float spline_z_grid_delta_inv,
    const int spline_num_splines,
    const int i,
    const int ix,
    const int iy,
    const int iz) 
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < spline_num_splines)
    eval_UBspline_3d_s_vgh ( 
        spline_coefs+ix*xs+iy*ys+iz*zs+n,
        xs, ys, zs, 
        walkers_vals+i*NSIZE+n,
        walkers_grads+i*MSIZE+n*3,
        walkers_hess+i*OSIZE+n*9,
        a,
        b,
        c,
        da,
        db,
        dc,
        d2a,
        d2b,
        d2c,
        spline_x_grid_delta_inv,
        spline_y_grid_delta_inv,
        spline_z_grid_delta_inv );
}
