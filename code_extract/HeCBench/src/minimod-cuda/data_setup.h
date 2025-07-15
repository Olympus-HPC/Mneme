#ifndef DATA_SETUP_H
#define DATA_SETUP_H

#include "constants.h"
#include "grid.h"

void target_init(struct grid_t grid, uint nsteps,
                 const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ phi,
                 const float *__restrict__ eta, const float *__restrict__ coefx, const float *__restrict__ coefy,
                 const float *__restrict__ coefz, const float *__restrict__ vp, const float *__restrict__ source);

void target(uint nsteps, double *time_kernel,
            struct grid_t grid,
            llint sx, llint sy, llint sz,
            float hdx_2, float hdy_2, float hdz_2,
            const float *__restrict__ coefx, const float *__restrict__ coefy, const float *__restrict__ coefz,
            float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ vp,
            const float *__restrict__ phi, const float *__restrict__ eta, const float *__restrict__ source);

void target_finalize(struct grid_t grid, uint nsteps,
                     const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ phi,
                     const float *__restrict__ eta, const float *__restrict__ coefx, const float *__restrict__ coefy,
                     const float *__restrict__ coefz, const float *__restrict__ vp, const float *__restrict__ source);

void kernel_add_source(struct grid_t grid,
                       float *__restrict__ u, const float *__restrict__ source, llint istep,
                       llint sx, llint sy, llint sz);

void find_min_max_u(struct grid_t grid,
                    const float *__restrict__ u, float *__restrict__ min_u, float *__restrict__ max_u);

#endif
