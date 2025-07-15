#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "constants.h"
#include "grid.h"
#include "pml.h"
#include "data_setup.h"

void init_coef(float dx, float *__restrict__ coefx)
{
  float dx2 = dx*dx;
  coefx[0] = -205.f/72.f/dx2;
  coefx[1] = 8.f/5.f/dx2;
  coefx[2] = -1.f/5.f/dx2;
  coefx[3] = 8.f/315.f/dx2;
  coefx[4] = -1.f/560.f/dx2;
}

float compute_dt_sch(const float *coefx, const float *coefy, const float *coefz)
{

  float ftmp = 0.f;
  ftmp += fabsf(coefx[0]) + fabsf(coefy[0]) + fabsf(coefz[0]);
  for (uint i = 1; i < 5; i++) {
    ftmp += 2.f*fabsf(coefx[i]);
    ftmp += 2.f*fabsf(coefy[i]);
    ftmp += 2.f*fabsf(coefz[i]);
  }
  return 2.f*cfl/(sqrtf(ftmp)*vmax);
}

void gaussian_source(uint nt, float dt, float *__restrict__ source)
{
  const float sigma = 0.6f*_fmax;
  const float tau = 1.0f;
  const float scale = 8.0f;

  for (uint it = 1; it <= nt; ++it) {
    float t = dt*(it-1);
    source[it-1] = -2.f*scale*sigma
      *(sigma-2.f*sigma*scale*POW2(sigma*t-tau))
      *expf(-scale*POW2(sigma*t-tau));
  }
}

void write_io(struct grid_t grid, const float *u, uint istep)
{
    char filename_buf[32];
    snprintf(filename_buf, sizeof(filename_buf), "snapshot.it%u.n%llu.raw", istep, grid.nz);
    FILE *snapshot_file = fopen(filename_buf, "wb");
    for (llint i = 0; i < grid.nx; ++i) {
        for (llint j = 0; j < grid.ny; ++j) {
            fwrite(&u[IDX3_grid(i,j,0,grid)], sizeof(float), grid.nz, snapshot_file);
        }
    }
    /* Clean up */
    fclose(snapshot_file);
}


int main(int argc, char *argv[])
{
  llint nx = 100, ny = 100, nz = 100;
  /* Task size (supported targets only) */
  llint tsx = 10, tsy = 10;
  uint nsteps = 1;
  uint niters = 1;
  bool disable_warm_up_iter = true;
  bool finalio = false;

  /* Process arguments */
  if( argc == 1 )
    printf( "Usage:\n ./main --grid N --nsteps M --finalio\nUsing default values, no IO.\n\n" );
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i],"--grid") == 0) {
      ++i;
      nx = strtoll(argv[i],NULL,10);
      ny = nx;
      nz = nx;
    }
    else if (strcmp(argv[i],"--tsize") == 0) {
      ++i;
      tsx = strtoll(argv[i],NULL,10);
      tsy = tsx;
      printf("tsx, tsy = %lld, %lld\n", tsx, tsy);
    }
    else if (strcmp(argv[i],"--nsteps") == 0) {
      ++i;
      nsteps = strtoll(argv[i],NULL,10);
      printf("nsteps = %d\n", nsteps);
    }
    else if (strcmp(argv[i],"--niters") == 0) {
      ++i;
      niters = strtoll(argv[i],NULL,10);
      printf("niters = %d\n", niters);
    }
    else if (strcmp(argv[i],"--warm-up") == 0) {
      disable_warm_up_iter = false;
      printf("warm up iteration is enabled\n");
    }
    else if (strcmp(argv[i],"--finalio") == 0) {
      finalio = true;
      printf("writing final wavefile is enabled\n");
    }
  }

  double total_kernel_time = 0.0;
  double total_modeling_time = 0.0;
  bool warm_up_iter = !disable_warm_up_iter;

  for (uint iiter = 0; iiter < (disable_warm_up_iter ? niters : niters+1); iiter++) {

    const struct grid_t grid = init_grid(nx,ny,nz,tsx,tsy);

    printf("grid = %lld %lld %lld\n", grid.nx, grid.ny, grid.nz);

    const llint sx = nx/2, sy = ny/2, sz = nz/2;

    float *u = allocateHostGrid (grid);
    float *v = allocateHostGrid (grid);
    float *phi = allocateHostGrid (grid);
    float *eta = allocateHostGrid (grid);
    float *vp = allocateHostGrid (grid);

    // Wave field initialization
    memset (u, 0, gridSize (grid));
    memset (v, 0, gridSize (grid));

    float *source = (float*) malloc(sizeof(float)*nsteps);

    float coefx[5], coefy[5], coefz[5];
    init_coef(grid.dx, coefx);
    init_coef(grid.dy, coefy);
    init_coef(grid.dz, coefz);

    const float dt_sch = compute_dt_sch(coefx, coefy, coefz);
    // Init vp and phi
    const float vp_all = POW2(2000.f*dt_sch);
    for (llint i = 0; i < nx; ++i) {
      for (llint j = 0; j < ny; ++j) {
        for (llint k = 0; k < nz; ++k) {
          phi[IDX3_grid(i,j,k,grid)] = 0.f;
          vp[IDX3_grid(i,j,k,grid)] = vp_all;
        }
      }
    }
    (void) gaussian_source(nsteps,dt_sch,source);
    // Init PML
    init_eta(grid, dt_sch, eta);

    const float hdx_2 = 1.f / (4.f * POW2(grid.dx));
    const float hdy_2 = 1.f / (4.f * POW2(grid.dy));
    const float hdz_2 = 1.f / (4.f * POW2(grid.dz));

    target_init(grid,nsteps,u,v,phi,eta,coefx,coefy,coefz,vp,source);

    // Time loop
    struct timespec start_m, end_m;
    clock_gettime(CLOCK_REALTIME, &start_m);

    double kernel_time;

    target(nsteps, &kernel_time,
           grid,
           sx, sy, sz,
           hdx_2, hdy_2, hdz_2,
           coefx, coefy, coefz,
           u, v, vp,
           phi, eta, source);

    if (warm_up_iter) {
      kernel_time = 0;
    }

    clock_gettime(CLOCK_REALTIME, &end_m);
    if (!warm_up_iter) {
      total_modeling_time += (end_m.tv_sec  - start_m.tv_sec) +
                             (double)(end_m.tv_nsec - start_m.tv_nsec) / 1.0e9;
      total_kernel_time += kernel_time;
    }

    float min_u, max_u;
    find_min_max_u(grid, u, &min_u, &max_u);

    printf("Checksum: min_u,  max_u = %f, %f\n", min_u, max_u);

    if( finalio ) write_io(grid, u, nsteps);

    target_finalize(grid,nsteps,u,v,phi,eta,coefx,coefy,coefz,vp,source);
    freeHostGrid(u, grid);
    freeHostGrid(v, grid);
    freeHostGrid(phi, grid);
    freeHostGrid(eta, grid);
    freeHostGrid(vp, grid);
    free(source);

    warm_up_iter = false;
  }

  printf("Average kernel time per iteration: %g s\n", total_kernel_time / niters);
  printf("Average modeling time per iteration: %g s\n", total_modeling_time / niters);

  return 0;
}
