//define the data set size for a cubic volume
#define DATAXSIZE 256
#define DATAYSIZE 256
#define DATAZSIZE 256

//define the chunk sizes that each threadblock will work on
#define BLKXSIZE 16
#define BLKYSIZE 4
#define BLKZSIZE 4

__device__ double Laplacian(const double c[][DATAYSIZE][DATAXSIZE],
                            double dx, double dy, double dz, int x, int y, int z)
{
  int xp, xn, yp, yn, zp, zn;

  int nx = (int)DATAXSIZE - 1;
  int ny = (int)DATAYSIZE - 1;
  int nz = (int)DATAZSIZE - 1;

  xp = x+1;
  xn = x-1;
  yp = y+1;
  yn = y-1;
  zp = z+1;
  zn = z-1;

  if (xp > nx) xp = 0;
  if (yp > ny) yp = 0;
  if (zp > nz) zp = 0;
  if (xn < 0)  xn = nx;
  if (yn < 0)  yn = ny;
  if (zn < 0)  zn = nz;

  double cxx = (c[z][y][xp] + c[z][y][xn] - 2.0*c[z][y][x]) / (dx*dx);
  double cyy = (c[z][yp][x] + c[z][yn][x] - 2.0*c[z][y][x]) / (dy*dy);
  double czz = (c[zp][y][x] + c[zn][y][x] - 2.0*c[z][y][x]) / (dz*dz);

  return cxx + cyy + czz;
}

__device__ double GradientX(const double phi[][DATAYSIZE][DATAXSIZE], 
                            double dx, double dy, double dz, int x, int y, int z)
{
  int nx = (int)DATAXSIZE - 1;
  int xp = x+1;
  int xn = x-1;

  if (xp > nx) xp = 0;
  if (xn < 0)  xn = nx;

  return (phi[z][y][xp] - phi[z][y][xn]) / (2.0*dx);
}

__device__ double GradientY(const double phi[][DATAYSIZE][DATAXSIZE], 
                            double dx, double dy, double dz, int x, int y, int z)
{
  int ny = (int)DATAYSIZE - 1;
  int yp = y+1;
  int yn = y-1;

  if (yp > ny) yp = 0;
  if (yn < 0)  yn = ny;

  return (phi[z][yp][x] - phi[z][yn][x]) / (2.0*dy);
}

__device__ double GradientZ(const double phi[][DATAYSIZE][DATAXSIZE],
                            double dx, double dy, double dz, int x, int y, int z)
{
  int nz = (int)DATAZSIZE - 1;
  int zp = z+1;
  int zn = z-1;

  if (zp > nz) zp = 0;
  if (zn < 0)  zn = nz;

  return (phi[zp][y][x] - phi[zn][y][x]) / (2.0*dz);
}

__global__ void chemicalPotential(
    const double c[][DATAYSIZE][DATAXSIZE], 
    double mu[][DATAYSIZE][DATAXSIZE], 
    double dx,
    double dy,
    double dz,
    double gamma,
    double e_AA,
    double e_BB,
    double e_AB)
{
  unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

  if ((idx < DATAXSIZE) && (idy < DATAYSIZE) && (idz < DATAZSIZE)) {

    mu[idz][idy][idx] = 4.5 * ( ( c[idz][idy][idx] + 1.0 ) * e_AA + 
        ( c[idz][idy][idx] - 1 ) * e_BB - 2.0 * c[idz][idy][idx] * e_AB ) + 
      3.0 * c[idz][idy][idx] + c[idz][idy][idx] * c[idz][idy][idx] * c[idz][idy][idx] - 
      gamma * Laplacian(c,dx,dy,dz,idx,idy,idz);
  }
}

__device__ double freeEnergy(double c, double e_AA, double e_BB, double e_AB)
{
  return (((9.0 / 4.0) * ((c*c+2.0*c+1.0)*e_AA+(c*c-2.0*c+1.0)*e_BB+
          2.0*(1.0-c*c)*e_AB)) + ((3.0/2.0) * c * c) + ((3.0/12.0) * c * c * c * c));
}

__global__ void localFreeEnergyFunctional(
    const double c[][DATAYSIZE][DATAXSIZE],
    double f[][DATAYSIZE][DATAXSIZE], 
    double dx,
    double dy,
    double dz,
    double gamma,
    double e_AA,
    double e_BB,
    double e_AB)
{
  unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

  if ((idx < DATAXSIZE) && (idy < DATAYSIZE) && (idz < DATAZSIZE)) {

    f[idz][idy][idx] = freeEnergy(c[idz][idy][idx],e_AA,e_BB,e_AB) + (gamma / 2.0) * (
        GradientX(c,dx,dy,dz,idx,idy,idz) * GradientX(c,dx,dy,dz,idx,idy,idz) + 
        GradientY(c,dx,dy,dz,idx,idy,idz) * GradientY(c,dx,dy,dz,idx,idy,idz) + 
        GradientZ(c,dx,dy,dz,idx,idy,idz) * GradientZ(c,dx,dy,dz,idx,idy,idz));
  }
}

__global__ void cahnHilliard(
    double cnew[][DATAYSIZE][DATAXSIZE], 
    const double cold[][DATAYSIZE][DATAXSIZE], 
    const double mu[][DATAYSIZE][DATAXSIZE],
    double D,
    double dt,
    double dx,
    double dy,
    double dz)
{
  unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
  if ((idx < DATAXSIZE) && (idy < DATAYSIZE) && (idz < DATAZSIZE)) {
    cnew[idz][idy][idx] = cold[idz][idy][idx] + dt * D * Laplacian(mu,dx,dy,dz,idx,idy,idz);
  }
}

__global__ void Swap(double cnew[][DATAYSIZE][DATAXSIZE], double cold[][DATAYSIZE][DATAXSIZE])
{
  unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;
  double tmp;    

  if ((idx < DATAXSIZE) && (idy < DATAYSIZE) && (idz < DATAZSIZE)) {
    tmp = cnew[idz][idy][idx];
    cnew[idz][idy][idx] = cold[idz][idy][idx];
    cold[idz][idy][idx] = tmp;
  }
}

