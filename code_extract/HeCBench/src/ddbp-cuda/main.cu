/*
This function reconstruct the 3D volume from projections, based on
the Distance-Driven principle. It works by calculating the overlap
in X and Y axis of the volume and the detector boundaries.
The geometry is for DBT with half cone-beam. All parameters are set
in "ParameterSettings" code.

Reference:
- Branchless Distance Driven Projection and Backprojection,
Samit Basu and Bruno De Man (2006)
- GPU Acceleration of Branchless Distance Driven Projection and
Backprojection, Liu et al (2016)
- GPU-Based Branchless Distance-Driven Projection and Backprojection,
Liu et al (2017)
- A GPU Implementation of Distance-Driven Computed Tomography,
Ryan D. Wagner (2017)
---------------------------------------------------------------------
Copyright (C) <2019>  <Rodrigo de Barros Vimieiro>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Original author: Rodrigo de Barros Vimieiro
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

// thread block size
#define BLOCK_SIZE 256

// integration direction
#define integrateXcoord 1
#define integrateYcoord 0

__global__ void pad_projections_kernel(
    double* d_img,
    const int nDetXMap,
    const int nDetYMap,
    const int nElem,
    const int np)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nElem)
    d_img[(np*nDetYMap *nDetXMap) + (gid*nDetYMap)] = 0;
}

__global__ void map_boudaries_kernel(
    double* d_pBound,
    const int nElem,
    const double valueLeftBound,
    const double sizeElem,
    const double offset)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nElem)
    d_pBound[gid] = (gid - valueLeftBound) * sizeElem + offset;
}

__global__ void rot_detector_kernel(
          double* __restrict__ d_pRdetY,
          double* __restrict__ d_pRdetZ,
    const double* __restrict__ d_pYcoord,
    const double* __restrict__ d_pZcoord,
    const double yOffset,
    const double zOffset,
    const double phi,
    const int nElem)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < nElem) {
    // cos and sin are in measured in radians.
    d_pRdetY[gid] = ((d_pYcoord[gid] - yOffset) * cos(phi) - 
                     (d_pZcoord[gid] - zOffset) * sin(phi)) + yOffset;
    d_pRdetZ[gid] = ((d_pYcoord[gid] - yOffset) * sin(phi) +
                     (d_pZcoord[gid] - zOffset) * cos(phi)) + zOffset;
  }
}

__global__ void mapDet2Slice_kernel(
           double* __restrict__ const pXmapp,
           double* __restrict__ const pYmapp,
    double tubeX,
    double tubeY,
    double tubeZ,
    const double* __restrict__ const pXcoord,
    const double* __restrict__ const pYcoord,
    const double* __restrict__ const pZcoord,
    const double* __restrict__ const pZSlicecoord,
    const int nDetXMap,
    const int nDetYMap,
    const int nz)
{
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;

  if (px < nDetYMap && py < nDetXMap) {

    const int pos = py * nDetYMap + px;

    pXmapp[pos] = ((pXcoord[py] - tubeX)*(pZSlicecoord[nz] - pZcoord[px]) - 
        (pXcoord[py] * tubeZ) + (pXcoord[py] * pZcoord[px])) / (-tubeZ + pZcoord[px]);

    if (py == 0)
      pYmapp[px] = ((pYcoord[px] - tubeY)*(pZSlicecoord[nz] - pZcoord[px]) -
          (pYcoord[px] * tubeZ) + (pYcoord[px] * pZcoord[px])) / (-tubeZ + pZcoord[px]);
  }
}

__global__ void img_integration_kernel(
    double* d_img,
    const int nPixX,
    const int nPixY,
    const bool direction,
    const int offsetX,
    const int offsetY,
    const int nSlices)
{
  /*
     Integration of 2D slices over the whole volume

     (S.1.Integration. - Liu et al(2017))

     Perform an inclusive scan
   */

  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int px = tx + offsetX;
  const int py = ty + offsetY;
  const int pz = blockIdx.z * blockDim.z + threadIdx.z;

  if (px >= nPixY || py >= nPixX || pz >= nSlices) return;

  if (direction == integrateXcoord) {

    for (int s = 1; s <= blockDim.y; s *= 2) {

      int spot = ty - s;

      double val = 0;

      if (spot >= 0) {
        val = d_img[(pz*nPixY*nPixX) + (offsetY + spot) * nPixY + px];
      }

      if (spot >= 0) {
        d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
      }
    }
  }
  else
  {
    for (int s = 1; s <= blockDim.x; s *= 2) {

      int spot = tx - s;

      double val = 0;

      if (spot >= 0) {
        val = d_img[(pz*nPixY*nPixX) + py * nPixY + spot + offsetX];
      }

      if (spot >= 0) {
        d_img[(pz*nPixY*nPixX) + (py * nPixY) + px] += val;
      }
    }
  }
}

__global__ void bilinear_interpolation_kernel(
          double* __restrict__ d_sliceI,
    const double* __restrict__ d_pProj,
    const double* __restrict__ d_pObjX,
    const double* __restrict__ d_pObjY,
    const double* __restrict__ d_pDetmX,
    const double* __restrict__ d_pDetmY,
    const int nPixXMap,
    const int nPixYMap,
    const int nDetXMap,
    const int nDetYMap,
    const int nDetX,
    const int nDetY,
    const int np) 
{
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;

  // Make sure we don't try and access memory outside the detector
  // by having any threads mapped there return early
  if (px >= nPixYMap || py >= nPixXMap) return;

  //  S.2. Interpolation - Liu et al (2017)

  // Adjust the mapped coordinates to cross the range of (0-nDetX).*duMap 
  // Divide by pixelSize to get a unitary pixel size
  const double xNormData = nDetX - d_pObjX[py] / d_pDetmX[0];
  const int    xData = floor(xNormData);
  const double alpha = xNormData - xData;

  // Adjust the mapped coordinates to cross the range of (0-nDetY).*dyMap  
  // Divide by pixelSize to get a unitary pixel size
  const double yNormData = (d_pObjY[px] / d_pDetmX[0]) - (d_pDetmY[0] / d_pDetmX[0]);
  const int    yData = floor(yNormData);
  const double beta = yNormData - yData;

  double d00, d01, d10, d11;
  if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY)) 
    d00 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData)];
  else
    d00 = 0.0;

  if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yNormData) >= 0) && ((yNormData) <= nDetY))
    d10 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData)];
  else
    d10 = 0.0;

  if (((xNormData) >= 0) && ((xNormData) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
    d01 = d_pProj[(np*nDetYMap*nDetXMap) + (xData*nDetYMap + yData + 1)];
  else
    d01 = 0.0;

  if (((xData + 1) > 0) && ((xData + 1) <= nDetX) && ((yData + 1) > 0) && ((yData + 1) <= nDetY))
    d11 = d_pProj[(np*nDetYMap*nDetXMap) + ((xData + 1)*nDetYMap + yData + 1)];
  else
    d11 = 0.0;

  double result_temp1 = alpha * d10 + (-d00 * alpha + d00);
  double result_temp2 = alpha * d11 + (-d01 * alpha + d01);

  d_sliceI[py * nPixYMap + px] = beta * result_temp2 + (-result_temp1 * beta + result_temp1);
}

__global__ void differentiation_kernel(
          double* __restrict__ d_pVolume,
    const double* __restrict__ d_sliceI,
    double tubeX,
    double rtubeY,
    double rtubeZ,
    const double* __restrict__ const d_pObjX,
    const double* __restrict__ const d_pObjY,
    const double* __restrict__ const d_pObjZ,
    const int nPixX,
    const int nPixY,
    const int nPixXMap,
    const int nPixYMap,
    const double du,
    const double dv,
    const double dx,
    const double dy,
    const double dz,
    const int nz) 
{
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;

  /*
     S.3. Differentiation - Eq. 24 - Liu et al (2017)

     Detector integral projection
     ___________
     |_A_|_B_|___|
     |_C_|_D_|___|
     |___|___|___|


     (px,py)
     ________________
     |_A_|__B__|_____|
     |_C_|(0,0)|(0,1)|
     |___|(1,0)|(1,1)|

     Threads are lauched from D up to nPixX (py) and nPixY (px)
     i.e., they are running on the detector image. Thread (0,0) is on D.

     Coordinates on intergal projection:

     A = py * nPixYMap + px
     B = ((py+1) * nPixYMap) + px
     C = py * nPixYMap + px + 1
     D = ((py+1) * nPixYMap) + px + 1
   */

  if (px < nPixY && py < nPixX) {

    const int pos = (nPixX*nPixY*nz) + (py * nPixY) + px;

    int coordA = py * nPixYMap + px;
    int coordB = ((py + 1) * nPixYMap) + px;
    int coordC = coordA + 1;
    int coordD = coordB + 1;

    // x - ray angle in X coord
    double gamma = atan((d_pObjX[py] + (dx / 2.0) - tubeX) / (rtubeZ - d_pObjZ[nz]));

    // x - ray angle in Y coord
    double alpha = atan((d_pObjY[px] + (dy / 2.0) - rtubeY) / (rtubeZ - d_pObjZ[nz]));

    double dA, dB, dC, dD;

    dA = d_sliceI[coordA];
    dB = d_sliceI[coordB];
    dC = d_sliceI[coordC];
    dD = d_sliceI[coordD];

    // Treat border of interpolated integral detector
    if (dC == 0 && dD == 0) {
      dC = dA;
      dD = dB;
    }

    // S.3.Differentiation - Eq. 24 - Liu et al(2017)
    d_pVolume[pos] += ((dD - dC - dB + dA)*(du*dv*dz / (cos(alpha)*cos(gamma)*dx*dy)));
  }
}

__global__ void division_kernel(
    double* d_img,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nProj)
{
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  const int pz = blockIdx.z * blockDim.z + threadIdx.z;

  if (px < nPixY && py < nPixX && pz < nSlices) {
    const int pos = (nPixX*nPixY*pz) + (py * nPixY) + px;
    d_img[pos] /= (double) nProj;
  }
}


// Branchless distance-driven backprojection 
void backprojectionDDb(
          double* const h_pVolume,
    const double* const h_pProj,
    const double* const h_pTubeAngle,
    const double* const h_pDetAngle,
    const int idXProj,
    const int nProj,
    const int nPixX,
    const int nPixY,
    const int nSlices,
    const int nDetX,
    const int nDetY,
    const double dx,
    const double dy,
    const double dz,
    const double du,
    const double dv,
    const double DSD,
    const double DDR,
    const double DAG)
{
  // Number of mapped detectors
  const int nDetXMap = nDetX + 1;
  const int nDetYMap = nDetY + 1;

  // Number of mapped pixels
  const int nPixXMap = nPixX + 1;
  const int nPixYMap = nPixY + 1;

  double *d_pProj, *d_sliceI, *d_pVolume;

  cudaMalloc((void **)&d_pProj, nDetXMap*nDetYMap*nProj * sizeof(double)); 
  cudaMalloc((void **)&d_sliceI, nPixXMap*nPixYMap * sizeof(double));
  cudaMalloc((void **)&d_pVolume, nPixX*nPixY*nSlices * sizeof(double));

  // device memory for projections coordinates
  double *d_pDetX, *d_pDetY, *d_pDetZ, *d_pObjX, *d_pObjY, *d_pObjZ;

  cudaMalloc((void **)&d_pDetX, nDetXMap * sizeof(double));
  cudaMalloc((void **)&d_pDetY, nDetYMap * sizeof(double));
  cudaMalloc((void **)&d_pDetZ, nDetYMap * sizeof(double));
  cudaMalloc((void **)&d_pObjX, nPixXMap * sizeof(double));
  cudaMalloc((void **)&d_pObjY, nPixYMap * sizeof(double));
  cudaMalloc((void **)&d_pObjZ, nSlices * sizeof(double));

  // device memory for mapped coordinates
  double *d_pDetmY, *d_pDetmX;

  cudaMalloc((void **)&d_pDetmY, nDetYMap * sizeof(double));
  cudaMalloc((void **)&d_pDetmX, nDetYMap * nDetXMap * sizeof(double));

  // device memory for rotated detector coords
  double *d_pRdetY, *d_pRdetZ;

  cudaMalloc((void **)&d_pRdetY, nDetYMap * sizeof(double));
  cudaMalloc((void **)&d_pRdetZ, nDetYMap * sizeof(double));

  auto start = std::chrono::steady_clock::now();

  // Will reuse grid configurations
  dim3 threadsPerBlock (1,1,1);
  dim3 blockSize (1,1,1);

  const int maxThreadsPerBlock = BLOCK_SIZE;

  // Copy projection data padding with zeros for image integation

  // Initialize first column and row with zeros
  const double* h_pProj_tmp;
  double* d_pProj_tmp;

  threadsPerBlock.x = maxThreadsPerBlock;
  blockSize.x = (nDetXMap / maxThreadsPerBlock) + 1;

  for (int np = 0; np < nProj; np++) {

    // Pad on X coord direction
    pad_projections_kernel <<<blockSize, threadsPerBlock>>> (d_pProj, nDetXMap, nDetYMap, nDetXMap, np);

    // Pad on Y coord direction
    d_pProj_tmp = d_pProj + (nDetXMap*nDetYMap*np) + 1;
    cudaMemset(d_pProj_tmp, 0, nPixY * sizeof(double));
  }

  // Copy projections data from host to device
  for (int np = 0; np < nProj; np++)
    for (int c = 0; c < nDetX; c++) {
      h_pProj_tmp = h_pProj + (c * nDetY) + (nDetX*nDetY*np);
      d_pProj_tmp = d_pProj + (((c + 1) * nDetYMap) + 1) + (nDetXMap*nDetYMap*np);
      cudaMemcpy(d_pProj_tmp, h_pProj_tmp, nDetY * sizeof(double), cudaMemcpyHostToDevice);
    }

  // Generate detector and object boudaries

  threadsPerBlock.x = maxThreadsPerBlock;

  blockSize.x = (nDetX / maxThreadsPerBlock) + 1;

  map_boudaries_kernel <<<blockSize, threadsPerBlock>>> (d_pDetX, nDetXMap, (double)nDetX, -du, 0.0);

  blockSize.x = (nDetY / maxThreadsPerBlock) + 1;

  map_boudaries_kernel <<<blockSize, threadsPerBlock>>> (d_pDetY, nDetYMap, nDetY / 2.0, dv, 0.0);

  blockSize.x = (nPixX / maxThreadsPerBlock) + 1;

  map_boudaries_kernel <<<blockSize, threadsPerBlock>>> (d_pObjX, nPixXMap, (double)nPixX, -dx, 0.0);

  blockSize.x = (nPixY / maxThreadsPerBlock) + 1;

  map_boudaries_kernel <<<blockSize, threadsPerBlock>>> (d_pObjY, nPixYMap, nPixY / 2.0, dy, 0.0);

  blockSize.x = (nSlices / maxThreadsPerBlock) + 1;

  map_boudaries_kernel <<<blockSize, threadsPerBlock>>> (d_pObjZ, nSlices, 0.0, dz, DAG + (dz / 2.0));

  // Initiate variables value with 0
  cudaMemset(d_pDetZ, 0, nDetYMap * sizeof(double));
  cudaMemset(d_pVolume, 0, nPixX * nPixY * nSlices * sizeof(double));

  // X - ray tube initial position
  double tubeX = 0;
  double tubeY = 0;
  double tubeZ = DSD;

  // Iso - center position
  double isoY = 0;
  double isoZ = DDR;

  // Integration of 2D projection over the whole projections
  // (S.1.Integration. - Liu et al(2017))

  // Naive integration o the X coord
  threadsPerBlock.x = 8;
  threadsPerBlock.y = 4;
  threadsPerBlock.z = 8;

  blockSize.x = (int)ceilf((float)nDetYMap / (threadsPerBlock.x - 1));
  blockSize.y = 1;
  blockSize.z = (int)ceilf((float)nProj / threadsPerBlock.z);

  int Xk = (int)ceilf((float)nDetXMap / (threadsPerBlock.x - 1));
  for (int k = 0; k < Xk; k++) {

    img_integration_kernel <<<blockSize, threadsPerBlock>>> (
        d_pProj, nDetXMap, nDetYMap, integrateXcoord, 0, k * 9, nProj);
  }

  // Naive integration o the Y coord
  threadsPerBlock.x = 4;
  threadsPerBlock.y = 8;
  threadsPerBlock.z = 8;

  blockSize.x = 1;
  blockSize.y = (int)ceilf((float)nDetXMap / (threadsPerBlock.y - 1));
  blockSize.z = (int)ceilf((float)nProj / threadsPerBlock.z);

  int Yk = (int)ceilf((float)nDetYMap / (threadsPerBlock.y - 1));
  for (int k = 0; k < Yk; k++) {

    img_integration_kernel <<<blockSize, threadsPerBlock>>> (
        d_pProj, nDetXMap, nDetYMap, integrateYcoord, k * 9, 0, nProj);
  }

  double* d_pDetmX_tmp = d_pDetmX + (nDetYMap * (nDetXMap-2));

  int projIni, projEnd, nProj2Run;
  if (idXProj == -1) {
    projIni = 0;
    projEnd = nProj;
    nProj2Run = nProj;
  }
  else {
    projIni = idXProj;
    projEnd = idXProj + 1;
    nProj2Run = 1;
  }

  // For each projection
  for (int p = projIni; p < projEnd; p++) {

    // Get specif tube angle for the projection
    double theta = h_pTubeAngle[p] * M_PI / 180.0;

    // Get specif detector angle for the projection
    double phi = h_pDetAngle[p] * M_PI / 180.0;

    //printf("Tube angle:%f Det angle:%f\n", theta, phi);

    // Tube rotation
    double rtubeY = ((tubeY - isoY)*cos(theta) - (tubeZ - isoZ)*sin(theta)) + isoY;
    double rtubeZ = ((tubeY - isoY)*sin(theta) + (tubeZ - isoZ)*cos(theta)) + isoZ;

    //printf("R tube Y:%f R tube Z:%f\n", rtubeY, rtubeZ);

    // Detector rotation
    threadsPerBlock.x = maxThreadsPerBlock;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blockSize.x = (nDetYMap / maxThreadsPerBlock) + 1;
    blockSize.y = 1;
    blockSize.z = 1;

    rot_detector_kernel <<<blockSize, threadsPerBlock>>> (
        d_pRdetY, d_pRdetZ, d_pDetY, d_pDetZ, isoY, isoZ, phi, nDetYMap);

    threadsPerBlock.x = 16;
    threadsPerBlock.y = 16;
    threadsPerBlock.z = 1;

    // For each slice
    for (int nz = 0; nz < nSlices; nz++) {

      // Map detector onto XY plane(Inside proj loop in case detector rotates)

      blockSize.x = (nDetYMap / threadsPerBlock.x) + 1;
      blockSize.y = (nDetXMap / threadsPerBlock.y) + 1;
      blockSize.z = 1;

      mapDet2Slice_kernel <<<blockSize, threadsPerBlock>>> (
          d_pDetmX, d_pDetmY, tubeX, rtubeY, rtubeZ, d_pDetX,
          d_pRdetY, d_pRdetZ, d_pObjZ, nDetXMap, nDetYMap, nz);

      //  S.2. Interpolation - Liu et al (2017)

      blockSize.x = (nPixYMap / threadsPerBlock.x) + 1;
      blockSize.y = (nPixXMap / threadsPerBlock.y) + 1;

      bilinear_interpolation_kernel <<<blockSize, threadsPerBlock>>> (
          d_sliceI, d_pProj, d_pObjX, d_pObjY, d_pDetmX_tmp, d_pDetmY,
          nPixXMap, nPixYMap, nDetXMap, nDetYMap, nDetX, nDetY, p);

      // S.3. Differentiation - Eq. 24 - Liu et al (2017)

      blockSize.x = (nPixY / threadsPerBlock.x) + 1;
      blockSize.y = (nPixX / threadsPerBlock.y) + 1;

      differentiation_kernel <<<blockSize, threadsPerBlock>>> (
          d_pVolume, d_sliceI, tubeX, rtubeY, rtubeZ, d_pObjX, d_pObjY, d_pObjZ,
          nPixX, nPixY, nPixXMap, nPixYMap, du, dv, dx, dy, dz, nz);

    } // Loop end slices

  } // Loop end Projections


  // Normalize volume dividing by the number of projections
  threadsPerBlock.x = 8;
  threadsPerBlock.y = 8;
  threadsPerBlock.z = 4;

  blockSize.x = (nPixY / threadsPerBlock.x) + 1;
  blockSize.y = (nPixX / threadsPerBlock.y) + 1;
  blockSize.z = (nSlices / threadsPerBlock.z) + 1;

  division_kernel <<<blockSize, threadsPerBlock>>> (d_pVolume, nPixX, nPixY, nSlices, nProj2Run);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution %f (s)\n", time * 1e-9f);

  cudaMemcpy(h_pVolume, d_pVolume, nSlices* nPixX * nPixY * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_pProj);
  cudaFree(d_sliceI);
  cudaFree(d_pVolume);
  cudaFree(d_pDetX);
  cudaFree(d_pDetY);
  cudaFree(d_pDetZ);
  cudaFree(d_pObjX);
  cudaFree(d_pObjY);
  cudaFree(d_pObjZ);
  cudaFree(d_pDetmY);
  cudaFree(d_pDetmX);
  cudaFree(d_pRdetY);
  cudaFree(d_pRdetZ);
}

int main() 
{
                            // image voxel density
  const int nPixX = 1996;   // number of voxels
  const int nPixY = 2457;   // number of voxels
  const int nSlices = 78;  

                            // detector panel pixel density
  const int nDetX = 1664;   // number of pixels
  const int nDetY = 2048;   // number of pixels

  const int nProj = 15;     // number of projections
  const int idXProj = -1;   // loop over all projections

  const double dx = 0.112;  // single voxel size (mm)
  const double dy = 0.112;
  const double dz = 1.0;

  const double du = 0.14;   // single detector size (mm)
  const double dv = 0.14;

  const double DSD = 700;   // distance from source to detector (mm)
  const double DDR = 0.0;   // distance from detector to pivot (mm)
  const double DAG = 25.0;  // distance of air gap (mm)

  const size_t pixVol = nPixX * nPixY * nSlices;
  const size_t detVol = nDetX * nDetY * nProj;
  double *h_pVolume = (double*) malloc (pixVol * sizeof(double));
  double *h_pProj = (double*) malloc (detVol * sizeof(double));

  double *h_pTubeAngle = (double*) malloc (nProj * sizeof(double));
  double *h_pDetAngle = (double*) malloc (nProj * sizeof(double));
  
  // tube angles in degrees
  for (int i = 0; i < nProj; i++) 
    h_pTubeAngle[i] = -7.5 + i * 15.0/nProj;

  // detector angles in degrees
  for (int i = 0; i < nProj; i++) 
    h_pDetAngle[i] = -2.1 + i * 4.2/nProj;

  // random values
  srand(123);
  for (size_t i = 0; i < pixVol; i++) 
    h_pVolume[i] = (double)rand() / (double)RAND_MAX;

  for (size_t i = 0; i < detVol; i++) 
    h_pProj[i] = (double)rand() / (double)RAND_MAX;

  backprojectionDDb(
    h_pVolume,
    h_pProj,
    h_pTubeAngle,
    h_pDetAngle,
    idXProj,
    nProj,
    nPixX, nPixY,
    nSlices,
    nDetX, nDetY,
    dx, dy, dz,
    du, dv,
    DSD, DDR, DAG);

  double checkSum = 0;
  for (size_t i = 0; i < pixVol; i++)
    checkSum += h_pVolume[i];
  printf("checksum = %lf\n", checkSum);

  free(h_pVolume);
  free(h_pTubeAngle);
  free(h_pDetAngle);
  free(h_pProj);
  return 0;
}
