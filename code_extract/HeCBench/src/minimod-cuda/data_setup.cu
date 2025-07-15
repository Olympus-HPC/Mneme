#include <float.h>
#include <math.h>
#include "data_setup.h"

void target_init(struct grid_t grid, uint nsteps,
                 const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ phi,
                 const float *__restrict__ eta, const float *__restrict__ coefx, const float *__restrict__ coefy,
                 const float *__restrict__ coefz, const float *__restrict__ vp, const float *__restrict__ source)
{
    // Nothing needed
}

void target_finalize(struct grid_t grid, uint nsteps,
                     const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ phi,
                     const float *__restrict__ eta, const float *__restrict__ coefx, const float *__restrict__ coefy,
                     const float *__restrict__ coefz, const float *__restrict__ vp, const float *__restrict__ source)
{
    // Nothing needed
}

void kernel_add_source(struct grid_t grid,
                       float *__restrict__ u, const float *__restrict__ source, llint istep,
                       llint sx, llint sy, llint sz)
{
    // Nothing needed
}

void find_min_max_u(struct grid_t grid,
                    const float *__restrict__ u, float *__restrict__ min_u, float *__restrict__ max_u)
{
    const llint nx = grid.nx;
    const llint ny = grid.ny;
    const llint nz = grid.nz;
    const llint lx = grid.lx;
    const llint ly = grid.ly;
    const llint lz = grid.lz;

    llint u_size = (nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);

   float min_val = FLT_MAX;
   #pragma omp parallel for reduction(min: min_val)
   for (llint i = 0; i < u_size; i++) {
     min_val = fmin(u[i], min_val);
   }
   *min_u = min_val;

   float max_val = FLT_MIN;
   #pragma omp parallel for reduction(max: max_val)
   for (llint i = 0; i < u_size; i++) {
     max_val = fmax(u[i], max_val);
   }
   *max_u = max_val;
}
