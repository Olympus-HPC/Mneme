#include <stdio.h>
#include <cuda.h>
#include "grid.h"
#include "constants.h"

struct grid_t init_grid(llint nx, llint ny, llint nz, llint tsx, llint tsy)
{
    struct grid_t grid;
    grid.nx = nx; grid.ny = ny; grid.nz = nz;
    grid.dx = 20;  grid.dy = 20;  grid.dz = 20;
    grid.lx = 4; grid.ly = 4; grid.lz = 4;
    grid.ntaperx = 3; grid.ntapery = 3; grid.ntaperz = 3;

    // extending x-dim with extra two radius just so that boundary checking can be eliminated from the implementation
    grid.ldimx = nx + 4 * grid.lx;
    grid.ldimy = ny + 2 * grid.ly;
    // Padding the Z dimension to align on 128B, for grid size of 1000, this is effectively 1024
    grid.ldimz = ((nz + 2 * grid.lz + 31) / 32) * 32;

    printf("ldimx: %lld, ldimy: %lld, ldimz: %lld\n", grid.ldimx, grid.ldimy, grid.ldimz);

    const float lambdamax = vmax/_fmax;
    grid.ndampx = grid.ntaperx * lambdamax / grid.dx;
    grid.ndampy = grid.ntapery * lambdamax / grid.dy;
    grid.ndampz = grid.ntaperz * lambdamax / grid.dz;

    grid.x1 = 0;
    grid.x2 = grid.ndampx;
    grid.x3 = grid.ndampx;
    grid.x4 = grid.nx-grid.ndampx;
    grid.x5 = grid.nx-grid.ndampx;
    grid.x6 = grid.nx;

    grid.y1 = 0;
    grid.y2 = grid.ndampy;
    grid.y3 = grid.ndampy;
    grid.y4 = grid.ny-grid.ndampy;
    grid.y5 = grid.ny-grid.ndampy;
    grid.y6 = grid.ny;

    grid.z1 = 0;
    grid.z2 = grid.ndampz;
    grid.z3 = grid.ndampz;
    grid.z4 = grid.nz-grid.ndampz;
    grid.z5 = grid.nz-grid.ndampz;
    grid.z6 = grid.nz;

    grid.tsx = tsx;
    grid.tsy = tsy;
    grid.ntx = nx/tsx;
    grid.nty = ny/tsy;

    printf("ndamp = %lld %lld %lld\n", grid.ndampx, grid.ndampy, grid.ndampz);
    return grid;
}

// Lead padding needed to align element (0,0,ndampz) on 128B cache line
size_t getLeadpad (struct grid_t grid)
{
  int align32 = (grid.lz + grid.ndampz) & 31;
  return (align32 ? 32 - align32 : 0);
}

// Useful size of the grid, in bytes
size_t gridSize (struct grid_t grid)
{
  size_t size = grid.ldimx * grid.ldimy * grid.ldimz * sizeof (float);
  return size;
}

// Device grid, with lead padding
float * allocateDeviceGrid (struct grid_t grid)
{
  int leadpad = getLeadpad(grid);
  size_t size = gridSize(grid) + leadpad * sizeof (float);
  float *ptr;
  if (cudaMalloc ((void **)&ptr, size) != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed!");
      return NULL;
  }
  return (ptr + leadpad);
}

void freeDeviceGrid (float *ptr, struct grid_t grid)
{
  int leadpad = getLeadpad(grid);
  cudaFree (ptr - leadpad);
}

// Host grid, with lead padding
float * allocateHostGrid (struct grid_t grid)
{
  int leadpad = getLeadpad(grid);
  size_t size = gridSize(grid) + leadpad * sizeof (float);
  float *ptr;
  if (cudaMallocHost ((void **)&ptr, size) != cudaSuccess) {
      fprintf(stderr, "cudaMallocHost failed!");
      return NULL;
  }
  return (ptr + leadpad);
}

void freeHostGrid (float *ptr, struct grid_t grid)
{
  int leadpad = getLeadpad(grid);
  cudaFreeHost (ptr - leadpad);
}
