#ifndef CONSTANTS_H
#define CONSTANTS_H

#define POW2(x) ((x)*(x))
#define IDX3(i,j,k)((llint)((i + lx) * ldimy + j + ly) * (llint)ldimz + k + lz)
#define IDX3_grid(i, j, k, grid) (((i + grid.lx) * grid.ldimy + j + grid.ly) * grid.ldimz + k + grid.lz)

extern const float _fmax;
extern const float vmin;
extern const float vmax;
extern const float cfl;

typedef long long int llint;
typedef unsigned int uint;

#endif
