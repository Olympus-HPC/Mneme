/*
 * Copyright 2010 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.

  Double-precision floating point atomic add
 */
__device__ __forceinline__
double atomic_add(double *address, double val)
{
  // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
  unsigned long long *ptr = (unsigned long long *)address;
  unsigned long long old, newdbl, ret = *ptr;
  do {
    old = ret;
    newdbl = __double_as_longlong(__longlong_as_double(old)+val);
  } while((ret = atomicCAS(ptr, old, newdbl)) != old);
  return __longlong_as_double(ret);
}

__global__
void compute_flux_x (const double *__restrict__ state,
                           double *__restrict__ flux,
                     const double *__restrict__ hy_dens_cell,
                     const double *__restrict__ hy_dens_theta_cell,
                     const double hv_coef,
                     const int nx,
                     const int nz,
                     const int hs)
{
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];

  if (i < nx+1 && k < nz) { 
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (int ll=0; ll<NUM_VARS; ll++) {
      for (int s=0; s < sten_size; s++) {
        int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+s;
        stencil[s] = state[inds];
      }
      //Fourth-order-accurate interpolation of the state
      vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
      //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
      d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    double r = vals[ID_DENS] + hy_dens_cell[k+hs];
    double u = vals[ID_UMOM] / r;
    double w = vals[ID_WMOM] / r;
    double t = ( vals[ID_RHOT] + hy_dens_theta_cell[k+hs] ) / r;
    double p = C0*pow((r*t),gamm);

    //Compute the flux vector
    flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u     - hv_coef*d3_vals[ID_DENS];
    flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*u+p - hv_coef*d3_vals[ID_UMOM];
    flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*w   - hv_coef*d3_vals[ID_WMOM];
    flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*u*t   - hv_coef*d3_vals[ID_RHOT];
  }
}

__global__
void compute_tend_x (const double *__restrict__ flux,
                           double *__restrict__ tend,
                     const int nx,
                     const int nz,
                     const int dx )
{
  int ll = blockIdx.z * blockDim.z + threadIdx.z;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx && k < nz) { 
    int indt  = ll* nz   * nx    + k* nx    + i  ;
    int indf1 = ll*(nz+1)*(nx+1) + k*(nx+1) + i  ;
    int indf2 = ll*(nz+1)*(nx+1) + k*(nx+1) + i+1;
    tend[indt] = -( flux[indf2] - flux[indf1] ) / dx;
  }
}

__global__
void compute_flux_z (const double *__restrict__ state,
                           double *__restrict__ flux,
                           double *__restrict__ hy_dens_int,
                           double *__restrict__ hy_pressure_int,
                           double *__restrict__ hy_dens_theta_int,
                     const double hv_coef,
                     const int nx,
                     const int nz,
                     const int hs)
{
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS];

  if (i < nx && k < nz+1) { 
    //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
    for (int ll=0; ll<NUM_VARS; ll++) {
      for (int s=0; s<sten_size; s++) {
        int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+s)*(nx+2*hs) + i+hs;
        stencil[s] = state[inds];
      }
      //Fourth-order-accurate interpolation of the state
      vals[ll] = -stencil[0]/12 + 7*stencil[1]/12 + 7*stencil[2]/12 - stencil[3]/12;
      //First-order-accurate interpolation of the third spatial derivative of the state
      d3_vals[ll] = -stencil[0] + 3*stencil[1] - 3*stencil[2] + stencil[3];
    }

    //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
    double r = vals[ID_DENS] + hy_dens_int[k];
    double u = vals[ID_UMOM] / r;
    double w = vals[ID_WMOM] / r;
    double t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r;
    double p = C0*pow((r*t),gamm) - hy_pressure_int[k];
    //Enforce vertical boundary condition and exact mass conservation
    if (k == 0 || k == nz) {
      w                = 0;
      d3_vals[ID_DENS] = 0;
    }

    //Compute the flux vector with hyperviscosity
    flux[ID_DENS*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w     - hv_coef*d3_vals[ID_DENS];
    flux[ID_UMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*u   - hv_coef*d3_vals[ID_UMOM];
    flux[ID_WMOM*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*w+p - hv_coef*d3_vals[ID_WMOM];
    flux[ID_RHOT*(nz+1)*(nx+1) + k*(nx+1) + i] = r*w*t   - hv_coef*d3_vals[ID_RHOT];
  }
}

__global__
void compute_tend_z (const double *__restrict__ state,
                     const double *__restrict__ flux,
                           double *__restrict__ tend,
                     const int nx,
                     const int nz,
                     const int dz )
{
  int ll = blockIdx.z * blockDim.z + threadIdx.z;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx && k < nz) { 
    int indt  = ll* nz   * nx    + k* nx    + i  ;
    int indf1 = ll*(nz+1)*(nx+1) + (k  )*(nx+1) + i;
    int indf2 = ll*(nz+1)*(nx+1) + (k+1)*(nx+1) + i;
    tend[indt] = -( flux[indf2] - flux[indf1] ) / dz;
    if (ll == ID_WMOM) {
      int inds = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
      tend[indt] = tend[indt] - state[inds]*grav;
    }
  }
}

__global__
void pack_send_buf (const double *__restrict__ state,
                          double *__restrict__ sendbuf_l,
                          double *__restrict__ sendbuf_r,
                    const int nx,
                    const int nz,
                    const int hs)
{
  int ll = blockIdx.z * blockDim.z + threadIdx.z;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < hs && k < nz) { 
    sendbuf_l[ll*nz*hs + k*hs + s] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + hs+s];
    sendbuf_r[ll*nz*hs + k*hs + s] = state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+s];
  }
}

__global__
void unpack_recv_buf (double *__restrict__ state,
                      const double *__restrict__ recvbuf_l,
                      const double *__restrict__ recvbuf_r,
                      const int nx,
                      const int nz,
                      const int hs)
{
  int ll = blockIdx.z * blockDim.z + threadIdx.z;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < hs && k < nz) { 
    state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + s      ] = recvbuf_l[ll*nz*hs + k*hs + s];
    state[ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + nx+hs+s] = recvbuf_r[ll*nz*hs + k*hs + s];
  }
}

__global__
void update_state_x (double *__restrict__ state,
                     const double *__restrict__ hy_dens_cell,
                     const double *__restrict__ hy_dens_theta_cell,
                     const int nx,
                     const int nz,
                     const int hs,
                     const int k_beg,
                     const double dz)
{
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hs && k < nz) { 
    double z = (k_beg + k+0.5)*dz;
    if (fabs(z-3*zlen/4) <= zlen/16) {
      int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
      int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
      int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i;
      state[ind_u] = (state[ind_r]+hy_dens_cell[k+hs]) * 50.;
      state[ind_t] = (state[ind_r]+hy_dens_cell[k+hs]) * 298. - hy_dens_theta_cell[k+hs];
    }
  }
}

__global__
void update_state_z (double *__restrict__ state,
                     const int data_spec_int,
                     const int i_beg,
                     const int nx,
                     const int nz,
                     const int hs,
                     const double dx,
                     const double mnt_width)
{
  int ll = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx+2*hs && ll < NUM_VARS) { 
    if (ll == ID_WMOM) {
      state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = 0.;
      state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = 0.;
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = 0.;
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = 0.;
      //Impose the vertical momentum effects of an artificial cos^2 mountain at the lower boundary
      if (data_spec_int == DATA_SPEC_MOUNTAIN) {
        double x = (i_beg+i-hs+0.5)*dx;
        if ( fabs(x-xlen/4) < mnt_width ) {
          double xloc = (x-(xlen/4)) / mnt_width;
          //Compute the derivative of the fake mountain
          double mnt_deriv = -pi*cos(pi*xloc/2)*sin(pi*xloc/2)*10/dx;
          //w = (dz/dx)*u
          state[ID_WMOM*(nz+2*hs)*(nx+2*hs) + (0)*(nx+2*hs) + i] = mnt_deriv*state[ID_UMOM*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
          state[ID_WMOM*(nz+2*hs)*(nx+2*hs) + (1)*(nx+2*hs) + i] = mnt_deriv*state[ID_UMOM*(nz+2*hs)*(nx+2*hs) + hs*(nx+2*hs) + i];
        }
      }
    } else {
      state[ll*(nz+2*hs)*(nx+2*hs) + (0      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
      state[ll*(nz+2*hs)*(nx+2*hs) + (1      )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (hs     )*(nx+2*hs) + i];
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs  )*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
      state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs+1)*(nx+2*hs) + i] = state[ll*(nz+2*hs)*(nx+2*hs) + (nz+hs-1)*(nx+2*hs) + i];
    }
  }
}

__global__
void acc_mass_te (double *__restrict__ mass,
                  double *__restrict__ te,
                  const double *__restrict__ state,
                  const double *__restrict__ hy_dens_cell,
                  const double *__restrict__ hy_dens_theta_cell,
                  const int nx,
                  const int nz,
                  const double dx,
                  const double dz)
{
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < nz && i < nx) {
    int ind_r = ID_DENS*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
    int ind_u = ID_UMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
    int ind_w = ID_WMOM*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
    int ind_t = ID_RHOT*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
    double r  = state[ind_r] + hy_dens_cell[hs+k];               // Density
    double u  = state[ind_u] / r;                                // U-wind
    double w  = state[ind_w] / r;                                // W-wind
    double th = ( state[ind_t] + hy_dens_theta_cell[hs+k] ) / r; // Potential Temperature (theta)
    double p  = C0*pow(r*th,gamm);                         // Pressure
    double t  = th / pow(p0/p,rd/cp);                      // Temperature
    double ke = r*(u*u+w*w);                                     // Kinetic Energy
    double ie = r*cv*t;                                          // Internal Energy

    // mass += r        *dx*dz; // Accumulate domain mass
    // te   += (ke + ie)*dx*dz; // Accumulate domain total energy
    atomic_add(mass, r*dx*dz);
    atomic_add(te, (ke+ie)*dx*dz);
  }
}

__global__
void update_fluid_state (const double *__restrict__ state_init,
                               double *__restrict__ state_out,
                         const double *__restrict__ tend,
                         const int nx,
                         const int nz,
                         const int hs,
                         const double dt)
{
  int ll = blockIdx.z * blockDim.z + threadIdx.z;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx && k < nz) { 
    int inds = ll*(nz+2*hs)*(nx+2*hs) + (k+hs)*(nx+2*hs) + i+hs;
    int indt = ll*nz*nx + k*nx + i;
    state_out[inds] = state_init[inds] + dt * tend[indt];
  }
}
