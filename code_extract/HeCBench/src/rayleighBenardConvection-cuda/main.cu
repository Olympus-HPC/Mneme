// NOTICE
// This file was stored by David Sanchez in 2023, based on a recovered hard-drive.  The code almost certainly dates to 2010 (earlier/later).
// I think I wrote this based on Matlab code by Grady Wright and Greg Barnett, under the supervision of Dave Yuen (and advice + comments from many).
// That said, I do not recall exactly when or under what conditions this was written and many parts may have been given to me or taken from other academic projects.
// In whole, in part, or in derivative this code forms the basis of some papers, but I've lost track of which ones.  I don't even know whether this
// is the most up-to-date such code.
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <chrono>
#include <cublas_v2.h>

// Parameters.  These may change from run-to-run
#define M         (500)         // vertical dimension of the temperature array
#define RA        (100000000.f) // Rayleigh number
#define XF        (2)           // Aspect ratio for the temperature array
#define STARTSTEP (0)           // First step

// Constants.  Stability cannot be assured if these values are altered.
//#define N (XF*(M-1)+1) // horizontal dimension of temperature array
#define N          (1000)
#define DX         (1.f/(M - 1.f)) // x and z-dimensional mesh spacing
#define DX2        (DX*DX) // DX^2, for saving a few computations.
#define OMEGACOEFF (-((DX*DX)*(DX*DX))*RA) // Used on every timestep

#define PI       3.1415926535897932384626433832795028841968 // Precision here is arbitrary and may be traded
                  // for performance if context allows
#define DT_START 0.000000000000005 //timestep width.  Needs to be small to
           //guide the model through high Ra runs.


#define FRAMESIZE (DX2/4.f) // How many iterations between saves
          // flatten a 2D grid with 1D blocks into a vector.  The functionality
          // could be extended to perform pointer arithmetic, but that's not necessary
          // in this code.
          //
          //  Invoked grid geometry
          //  (2D grid with 1D blocks)
          //
          //  |---|---|---|
          //  |1,1|1,2|1,3|
          //  |---|---|---|
          //  |2,1|2,2|2,3|  ==>  1,1 1,2 1,3 2,1 2,2 ...
          //  |---|---|---|
          //  |3,1|3,2|3,3|
          //  |---|---|---|


#define THREADID (((gridDim.x)*(blockIdx.y) + (blockIdx.x))*(blockDim.x) + (threadIdx.x))

// It is possible to alternate between 2D row-major and column-major formats by
// taking transposes.
#define TPOSE(i,j,ld) ((j)*(ld) + (i))

// Simplify calling G, since many of the arguments are assured
#define SHORTG(input, compute, save, frames) \
  G(h, input, d_Tbuff, d_DxT, d_y, d_u, d_v, d_psi, d_omega, d_dsc, \
    d_dsr, d_ei, d_dt, save, h_T, compute, frames, tstep)

//=============================================================================
//                  KERNELS
//=============================================================================

//=============================================================================
//                 ElementMultOmega
//=============================================================================
// Performs elementwise matrix multiplication on matrices shaped like omega,
// returning the result in A.
__global__ void ElemMultOmega(float* A, const float* B) {
  if(THREADID < (M-2)*(N-2) ) {
    A[THREADID] = A[THREADID]*B[THREADID];
  }
}

//=============================================================================
//                 ElementMultT
//=============================================================================
// Performs elementwise matrix multiplication on matrices shaped like T,
// putting the result in A.
__global__ void ElemMultT(float* A, const float* B) {
  if(THREADID < (M-2)*(N)) {
    A[THREADID] = A[THREADID]*B[THREADID];
  }
}

//=============================================================================
//                 ElementMultNu
//=============================================================================
// Performs elementwise matrix multiplication on matrices shaped like d_nutop,
// putting the result in A.
__global__ void ElemMultNu(float* A, const float* B) {
  if(THREADID < N) {
    A[THREADID] = A[THREADID]*B[THREADID];
  }
}

//=============================================================================
//                    SubOne
//=============================================================================
// Subtracts 1.f from every element in a vector (floats) of length N
__global__ void SubOne(float* A) {
  if(THREADID < N) {
    A[THREADID] = A[THREADID] - 1.f;
  }
}

//=============================================================================
//                    AddOne
//=============================================================================
// Adds 1.f from every element in a vector (floats) of length N
__global__ void AddOne(float* A) {
  if(THREADID < N) {
    A[THREADID] = A[THREADID] + 1.f;
  }
}

//=============================================================================
//                    AddX
//=============================================================================
// Adds x from every element in a vector (floats) of length (M-2)*N
__global__ void AddX(float* A, const float x) {
  if(THREADID < (M-2)*N) {
    A[THREADID] = A[THREADID] + x;
  }
}

//=============================================================================
//                   Updatedt
//=============================================================================
// Adaptive update rule for dt.  d_dt (a device-side one-element array) should
// be passed as dt, whereas ptru and ptrv point (1-indexed) to vectors u and v.
// The current value of dt[0] will be overwritten. Can be called with a 1D
// grid containing a single 1D block with one thread.
__global__ void Updatedt(int ptru, const float*__restrict__ u, int ptrv,
                         const float*__restrict__ v, float* dt) {
  dt[0] = max(abs(u[ptru - 1]),abs(v[ptrv - 1]));
  dt[0] = min(DX/dt[0],DX2/4.f);
}


//=============================================================================
//                  NusseltCompute
//=============================================================================
// Returns the Nusselt number of the array T, which is pointed to in GPU space
float NusseltCompute(cublasHandle_t &h,
                     float* T, float* nutop, float* ztop,
                     float* zbot,float* nubot, float* trnu)
{
  float topsum, botsum;
  // Calculate the Nusselt number along the top of the array.
  // d_nutop is the last three rows of T, in inverse order,
  // with all 0s along the bottom.

  // Copy the last three rows of T into the first three rows of nutop
  cublasScopy(h, N, (T + (M-5)*N), 1, (nutop), 1);
  cublasScopy(h, N, (T + (M-4)*N), 1, (nutop + (N)), 1);
  cublasScopy(h, N, (T + (M-3)*N), 1, (nutop + (2*N)), 1);

  float alpha = 0.f;
  float result;

  // Set the last row of nutop = 0.
  cublasSscal(h, N, &alpha, (nutop + (3*N)), 1);

  // nutop += -( 1 - ztop)
  // => nutop += ztop; nutop -= 1
  alpha = 1.f;
  cublasSaxpy(h, 4*N, &alpha, ztop, 1, nutop, 1);
  // Subtract 1 from every element in the array.  SubOne works on rows.
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nutop);
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nutop + N);
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nutop + 2*N);
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nutop + 3*N);

  // -(2/3)*row0 + 3*row1 - 6*row2 + (11/3)*row3
  // accumulate in the 0th row
  // scale the 0th row by -(2/3)
  alpha = -(2.f/3.f);
  cublasSscal(h, N, &alpha, nutop, 1);
  // Add 3*row1
  alpha = 3.f;
  cublasSaxpy(h, N, &alpha, (nutop + N), 1, nutop, 1);
  // Add - 6*row2
  alpha = -6.f;
  cublasSaxpy(h, N, &alpha, (nutop + (2*N)), 1, nutop, 1);
  // Add (11/3)*row3
  alpha = (11.f/3.f);
  cublasSaxpy(h, N, &alpha, (nutop + (3*N)), 1, nutop, 1);
  // Divide the array by 2*DX
  alpha = 1.f/(2.f*DX);
  cublasSscal(h, 4*N, &alpha, nutop, 1);
  // Elementwise multiplication with trnu
  ElemMultNu<<<floorf(N/256.f) + 1, 256>>>(nutop, trnu);
  // Sum up the elements of row0, by performing a dot product with
  // a row that has been altered to be all 1s.
  // Empty row1, then add 1 to all its elements
  alpha = 0.f;
  cublasSscal(h, N, &alpha, (nutop + N), 1);
  AddOne<<<floorf(N/256.f) + 1, 256>>>(nutop + N);

  cublasSdot(h, N, nutop, 1, (nutop + N), 1, &result);
  topsum = result /(-XF);

  // Calculate the Nusselt number along the bottom of the array.
  // d_nubot's first row is all 1, and ith row is the i-1th row of d_T
  // Put the first row of T in nubot, then subtract to get 0, then AddOne
  cublasSscal(h, N, &alpha, nubot, 1);
  AddOne<<<floorf(N/256.f) + 1, 256>>>(nubot);
  cublasScopy(h, N, T, 1, (nubot + N), 1);
  cublasScopy(h, N, (T + N), 1, (nubot + 2*N), 1);
  cublasScopy(h, N, (T + 2*N), 1, (nubot + 3*N), 1);

  // nubot += -( 1 - zbot)
  // => nubot += zbot; nubot -= 1
  alpha = 1.f;
  cublasSaxpy(h, 4*N, &alpha, zbot, 1, nubot, 1);
  // Subtract 1 from every element in the array.  SubOne works on rows.
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nubot);
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nubot + N);
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nubot + 2*N);
  SubOne<<<floorf(N/256.f) + 1, 256>>>(nubot + 3*N);

  // -(11/3)*row0 + 6*row1 - 3*row2 + (2/3)*row3
  // accumulate in the 0th row
  // scale the 0th row by -(11/3)
  alpha = -(11.f/3.f);
  cublasSscal(h, N, &alpha, nubot, 1);
  // Add 6*row1
  alpha = 6.f;
  cublasSaxpy(h, N, &alpha, (nubot + N), 1, nubot, 1);
  // Add -3*row2
  alpha = -3.f;
  cublasSaxpy(h, N, &alpha, (nubot + (2*N)), 1, nubot, 1);
  // Add (2/3)*row3
  alpha = 2.f/3.f;
  cublasSaxpy(h, N, &alpha, (nubot + (3*N)), 1, nubot, 1);
  // Divide the array by 2*DX
  alpha = 1.f/(2.f*DX);
  cublasSscal(h, 4*N, &alpha, nubot, 1);
  // Elementwise multiplication with trnu
  ElemMultNu<<<floorf(N/256.f) + 1, 256>>>(nubot, trnu);
  // Sum up the elements of row0, by performing a dot product with
  // a row that has been altered to be all 1s.
  // The second row of nutop has already been set up for this.
  cublasSdot(h, N, nubot, 1, (nutop + N), 1, &result);
  botsum = result /(-XF);

  return(topsum);
}
//=============================================================================
//                    Dz
//=============================================================================
// Finite-difference approximation to the first derivative with respect to z of
// a matrix shaped like T.  Execution forks if f is known to be psi.  Uses only
// row manipulations and the subtraction of 1 from each element of a vector. To
// extract a row, cublas routines are used.  The elements of the first column
// are separated in memory by N elements, so the initial elements of each row
// are likewise separated.  The individual elements of a single row are
// separated in memory by 1 element.

void Dz(cublasHandle_t &h, float* f, int is_it_psi, float* y) {
  // yrows[i] = frows[i + 1] - frows[i - 1]
  // Move all but the first two rows of f into the interior rows of y.
  // The end of one row is one element away from the beginning of the next,
  // so adjacent rows are laid out in memory identically to a vector.
  cublasScopy(h, N*(M-4), (f + (2*N)), 1, (y + N), 1);
  // Subtract all but the last two rows of f.
  const float alpha = -1.f;
  const float alpha2 = 1.f/(2.f*DX);
  cublasSaxpy(h, N*(M-4), &alpha, f, 1, (y + N), 1);

  if(is_it_psi == 1) {
    // yrows[0] = frows[1]
    // Move the second row of f into the first row of y
    cublasScopy(h, N, (f + N), 1, y, 1);
  }
  else {
    // yrows[0] = frows[1] - 1
    // Move the second row of f into the first row of y
    cublasScopy(h, N, (f + N), 1, y, 1);
    // Subtract 1 from every element in the first row of y.
    SubOne<<<floorf(N/256.f) + 1, 256>>>(y);
  }

  // yrows[M-3] = -frows[M-4]
  // Copy the second-to-last row of f into the last row of y
  cublasScopy(h, N, (f + ((M - 4)*N)), 1, (y + (M - 3)*N), 1);
  // Scale by -1.f
  cublasSscal(h, N, &alpha, (y + (M - 3)*N), 1);
  // Scale y by 1/(2*DX)
  cublasSscal(h, N*(M-2), &alpha2, y, 1);
}

//=============================================================================
//                    Dzz
//=============================================================================
// Finite-difference approximation to the second derivative with respect to z
// of a  T-shaped array.  Uses only row manipulations and the addition of 1 to
// each element of a vector.  To extract a row, cublas routines are used.  The
// elements of the first column are separated in memory by N elements, so the
// initial elements of each row are likewise separated.  The individual
// elements of a single row are separated in memory by 1 element.

void Dzz(cublasHandle_t &h, float* f, float* y) {
  // yrows[i] = frows[i - 1] - 2*frows[i] + frows[i + 1]
  // Move all but the last two rows of f into the interior rows of y.
  cublasScopy(h, N*(M-4), f, 1, (y + N), 1);

  const float alpha = -2.f;
  const float alpha2 = 1.f;
  const float alpha3 = 1.f/DX2;

  // Subtract 2* the interior rows of f
  cublasSaxpy(h, N*(M-4), &alpha, (f + N), 1, (y + N), 1);

  // Add all but the first two rows of f.
  cublasSaxpy(h, N*(M-4), &alpha2, (f + (2*N)), 1, (y + N), 1);

  // yrows[0] = 1 - 2*frows[0] + frows[1]
  // Copy the first row of f into the first row of y
  cublasScopy(h, N, f, 1, y, 1);

  // scale by -2
  cublasSscal(h, N, &alpha, y, 1);

  // add the second row of f
  cublasSaxpy(h, N, &alpha2, (f + N), 1, y, 1);

  // Add 1 to every element in the first row of y.
  AddOne<<<floorf(N/256.f) + 1, 256>>>(y);

  // yrows[M-3] = frows[M-4] - 2*frows[M-3]
  // move the second-to-last row of f into the last row of y
  cublasScopy(h, N, (f + (M - 4)*N), 1, (y + (M - 3)*N), 1);
  // subtract -2* the last row of f
  cublasSaxpy(h, N, &alpha, (f + (M - 3)*N), 1, (y + (M - 3)*N), 1);
  // Scale y by 1/DX2
  cublasSscal(h, (M-2)*N, &alpha3, y, 1);
}

//=============================================================================
//                    Dx
//=============================================================================
// Finite difference approximation to the first derivative with
// respect to x of a T-shaped matrix.  Forks if f is known to be psi. Uses only
// column manipulations and assumes all matrices are in row-major.  To extract
// a column, cublas routines are used.  If the beginning of an array is at f,
// then the elements of the first row (start of each column) are separated by
// one element, and each element within a column is separated by the length
// of a row, N.

void Dx(cublasHandle_t &h, float* f, int is_it_psi, float* y) {
  // ycols[i] = fcols[i+1] - fcols[i-1], interior cols
  // Copy all but the first two columns of f into the interior columns of y.
  // Copy row-by-row instead of column-by-column, since Dcopy is optimized
  // for longer vectors.

  for(int i = 0; i < M - 2; i++) {
    cublasScopy(h, (N-2), (f + i*N) + 2, 1, (y + i*N) + 1, 1);
  }
  // Subtract the block corresponding to all but the last two columns of f.
  float alpha = -1.f;
  for(int i = 0; i < M - 2; i++) {
    cublasSaxpy(h, (N-2), &alpha, (f + i*N), 1, (y + i*N) + 1, 1);
  }

  if(is_it_psi == 1) {
    float alpha = 6.f;
    float alpha2 = -3.f;
    float alpha3 = 2.f/3.f;
    // ycols[0] = 6*fcols[1] - 3*fcols[2] + (2/3)*fcols[3]
    // Begin by copying the second column of f into the first column of y
    cublasScopy(h, (M-2), (f + 1), N, y, N);
    // Scale it by a factor of 6
    cublasSscal(h, (M-2), &alpha, y, N);
    // Subtract the third column of 3*f
    cublasSaxpy(h, (M-2), &alpha2, (f + 2), N, y, N);
    // Add the fourth column of (2/3)*f
    cublasSaxpy(h, (M-2), &alpha3, (f + 3), N, y, N);

    alpha = -alpha;
    alpha2 = -alpha2;
    alpha3 = -alpha3;
    //ycols[N-1] = -6*fcols[N-2] + 3*fcols[N-3] - (2/3)*fcols[N-4]
    // Copy the second-to-last column of f into the last column of y
    cublasScopy(h, (M-2), (f + (N - 2)), N, (y + (N - 1)), N);
    // Scale it by a factor of -6
    cublasSscal(h, (M-2), &alpha, (y + (N - 1)), N);
    // Add the third-to-last column of 3*fcols[N-3]
    cublasSaxpy(h, (M-2), &alpha2, (f + (N - 3)), N, (y + (N - 1)), N);
    // Subtract the fourth-to-last column of (2/3)*f[N-4]
    cublasSaxpy(h, (M-2), &alpha3, (f + (N - 4)), N, (y + (N - 1)), N);
  }
  else {
    // outside columns = 0
    const float alpha = 0.f;
    cublasSscal(h, (M-2), &alpha, y, N);
    cublasSscal(h, (M-2), &alpha, (y + (N - 1)), N);

  }

  // Scale y by (1/(2*DX))
  alpha = (1.f/(2.f*DX));
  cublasSscal(h, N*(M-2), &alpha, y, 1);
}

//=============================================================================
//                    Dxx
//=============================================================================
// Finite-difference approximation to the second derivative with
// respect to x of a T-shaped matrix.  The input is always going to be a temp-
// erature array.  Uses only column manipulations and assumes all matrices are
// in row-major.  To extract a column, cublas routines are used.  If the
// beginning of an array is at f, then the elements of the first row (start of
// each column) are separated by one element, and each element within a column
// is separated by the length of a row, N.

void Dxx(cublasHandle_t &h, float* f, float* y) {
  const float alpha = -2.f;
  const float alpha2 = 2.f;
  const float alpha3 = 1.f;
  const float alpha4 = 1.f/(DX2);

  // ycols[i] = fcols[i-1] - 2*fcols[i] + fcols[i+1], interior columns
  // copy f into y
  cublasScopy(h, N*(M-2), f, 1, y, 1);
  // scale by -2
  cublasSscal(h, N*(M-2), &alpha, y, 1);
  // Add the block corresponding to all but the last two columns of f.
  for(int i = 0; i < M-2; i++) {
    cublasSaxpy(h, (N-2), &alpha3, (f + i*N), 1, (y + i*N) + 1, 1);
  }
  // Add the block corresponding to all but the first two columns of f.
  for(int i = 0; i < M-2; i++) {
    cublasSaxpy(h, (N-2), &alpha3, (f + i*N) + 2, 1, (y + i*N) + 1, 1);
  }

  // ycols[0] = -2*fcols[0] + 2*fcols[1]
  // Copy the first column of f into the first column of y.
  cublasScopy(h, (M-2), f, N, y, N);
  // Scale the first column of y by -2.f
  cublasSscal(h, (M-2), &alpha, y, N);
  // Add 2* the second column of f.
  cublasSaxpy(h, (M-2), &alpha2, (f + 1), N, y, N);

  // ycols[N-1] = -2*fcols[N-1] + 2*fcols[N-2]
  // Move the last column of f into the last column of y
  cublasScopy(h, (M-2), (f + (N - 1)), N, (y + (N - 1)), N);
  // Scale by -2.f
  cublasSscal(h, (M-2), &alpha, (y + (N - 1)), N);
  // add 2* the second-to-last column of f.
  cublasSaxpy(h, (M-2), &alpha2, (f + (N - 2)), N, (y + (N - 1)), N);

  // Scale y by 1/(DX^2)
  cublasSscal(h, N*(M-2), &alpha4, y, 1);
}

//=============================================================================
//                     G
//=============================================================================
// Computes the RK1 approximation using finite difference method, storing the
// result in output
void G(cublasHandle_t &h,
       float* f,
       float* Tbuff,
       float* DxT,
       float* y,
       float* u,
       float* v,
       float* psi,
       float* omega,
       float* dsc,
       float* dsr,
       float* ei,
       float* dt,
       float* output,
       float* h_T,
       int compute_velocity,
       int frames,
       int tstep) 
{
  // Define the grid dimensions
  dim3 grid(floorf(N/16.f) + 1, floorf((M-2)/16.f) + 1), block(256);

  // Define omega to be the interior columns of Dxf
  // Save Dx of f in DxT for later
  cublasScopy(h, N*(M-2), f, 1, DxT, 1);
  Dx(h, f, 0, DxT);

  // Copy the interior columns of DxT to omega
  for(int i = 0; i < M-2; i ++) {
    cublasScopy(h, (N-2), (DxT + i*N)+1, 1, (omega + i*(N-2)), 1);
  }

  // Perform some matrix multiplications.  cublas assumes everything is in
  // column-major, so while we want to perform:
  // omega = dsc*omega
  // omega = omega*dsr
  // omega = omega.*ei
  // omega = dsc*omega
  // omega = omega*dsr
  // we observe that Transpose(A*B) = Transpose(B)*Transpose(A) to perform
  // these same manipulations while preserving row-major storage.
  // Omega has dimensions (M-2)xN, but cublas thinks this is Nx(M-2).
  // dsc and dsr are square..
  // Perform Transpose(omega) = Transpose(omega)*Transpose(dsc):
  //
  // The call for cublasSgemm is (CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb,
  // beta, C, ldc), where A is m-by-k, B is k-by-n, and C is m-by-n.
  // Since A = C, k = n, so B is k-by-k.  B = dsc, so k=n=M-2 and m = N-2.
  const float alpha = 1.f;
  const float alpha2 = -1.f;
  const float beta = 0.f;

  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N-2, M-2, M-2, &alpha, omega, N-2, dsc, M-2, &beta, Tbuff, N-2 );
  cublasScopy(h, (N-2)*(M-2), Tbuff, 1, omega, 1);

  // Perform Tranpose(dsr)*Transpose(omega), store in omega.
  // since A is square, M = k = N-2, so n must be M-2.
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N-2, M-2, N-2, &alpha, dsr, N-2, omega, N-2, &beta, Tbuff, N-2);
  cublasScopy(h, (N-2)*(M-2), Tbuff, 1, omega, 1);

  // elementwise matrix multiplication, storing the result in omega
  //DEBUG
  ElemMultOmega<<<grid, block>>>(omega, ei);

  // same Transpose(omega)*Transpose(dsc) operation as before
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N-2, M-2, M-2, &alpha, omega, N-2, dsc, M-2, &beta, Tbuff, N-2);
  cublasScopy(h, (N-2)*(M-2), Tbuff, 1, omega, 1);


  // same Transpose(dsr)*Transpose(omega) operation as before
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N-2, M-2, N-2, &alpha, dsr, N-2, omega, N-2, &beta, Tbuff, N-2);
  cublasScopy(h, (N-2)*(M-2), Tbuff, 1, omega, 1);

  // Scale omega by -(DX^4)*(RA) = OMEGACOEFF
  const float omegacoeff = OMEGACOEFF;
  cublasSscal(h, (N-2)*(M-2), &omegacoeff, omega, 1);

  // interior columns of psi = (RA*DX^4)*omega
  // copy omega into the interior columns of psi
  // omega has rows of length N-2 instead of N

  for(int i = 0; i < M-2; i ++) {
    cublasScopy(h, (N-2), (omega + i*(N-2)), 1, (psi + i*N)+1, 1);
  }

  // Velocity in the x-direction
  cublasScopy(h, N*(M-2), f, 1, u, 1);
  Dz(h, psi, 1, u);

  // v is -Dxpsi, velocity in the z direction.
  // Place Dxpsi into v
  cublasScopy(h, N*(M-2), f, 1, v, 1);
  Dx(h, psi, 1, v);

  // Change the sign of v.
  cublasSscal(h, (M-2)*N, &alpha2, v, 1);
  // Store v in the z velocity file

  // If compute_velocity = 1, we need to update dt
  if(compute_velocity == 1) {
    // CublasIdamax returns 1-indexed pointers into the max element of a
    // float-precision vector
    int iu, iv;
    cublasIsamax(h, N*(M-2), u, 1, &iu);
    cublasIsamax(h, N*(M-2), v, 1, &iv);
    Updatedt<<<1,1>>>(iu, u, iv, v, dt);
  }

  // Place Dxxf into y
  cublasScopy(h, N*(M-2), f, 1, y, 1);
  Dxx(h, f, y);

  // y = y + Dzzf
  // place Dzzf into Tbuff
  cublasScopy(h, N*(M-2), f, 1, Tbuff, 1);
  Dzz(h, f, Tbuff);

  // Add the elements of y and Tbuff, storing in y.
  cublasSaxpy(h, N*(M-2), &alpha, Tbuff, 1, y, 1);

  // u = u.*DxT, where .* denotes elementwise multiplication
  // Perform the elentwise multiplication, storing in u
  ElemMultT<<<grid, block>>>(u,DxT);

  // y = y + u
  // Add y and u, storing the result in y
  cublasSaxpy(h, N*(M-2), &alpha, u, 1, y, 1);

  // u = DzT
  cublasScopy(h, N*(M-2), f, 1, u, 1);
  Dz(h, f, 0, u);

  // u = v.*u, where .* denotes elementwise multiplication.
  ElemMultT<<<grid, block>>>(u,v);

  // y = y + u
  cublasSaxpy(h, N*(M-2), &alpha, u, 1, y, 1);

  // copy into output
  cublasScopy(h, N*(M-2), y, 1, output, 1);
}


//=============================================================================
//                 ENTRY POINT
//=============================================================================

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <timesteps>\n", argv[0]);
    return 1;
  }
  const int ENDSTEP = atoi(argv[1]); // Last timestep to compute

  printf("M = %d. ", M);   // vertical dimension of the temperature array
  printf("N = %d. ", N);   // horizontal dimension of temperature array
  printf("DX = %E. ", DX); // x and z-dimensional mesh spacing
  printf("Ra = %E.\n", RA);// Rayleigh number

  printf("\nInitialization\n");
  cublasHandle_t h;
  cublasStatus_t stat = cublasCreate(&h);
  if(stat != CUBLAS_STATUS_SUCCESS) {
    printf("Failed to initialize CUBLAS\n");
    return EXIT_FAILURE;
  }

  printf("Configuring host and device memory.\n");
  // initialize dt to the chosen start parameter
  float dt = DT_START;
  float h_dt = dt;

  float* d_dt; // Device-side shadow of dt, as a one-element array.
  cudaMalloc((void**)&d_dt, sizeof(float));
  cublasSetVector(1, sizeof(float), &h_dt, 1, d_dt, 1);

  // Define T
  float* d_T;
  cudaMalloc((void**)&d_T, N*(M-2) * sizeof(float));

  float* h_X = (float*)malloc(N*sizeof(float));
  float* h_Z = (float*)malloc((M-2)*sizeof(float));
  float* h_T = (float*)malloc(N*(M-2)*sizeof(float));

  // Set the values of X and Z
  for(int i = 0; i < N; i++) {
    h_X[i] = (i*XF + 0.f)/(N - 1.f);
  }

  // Some wonky indexing here to accomodate the fact that T has its top and
  // bottom rows chopped off.
  for(int i = 1; i < (M - 1); i++) {
    h_Z[i - 1] = (i + 0.f)/(M - 1.f);
  }
  // Initialize T, perturbing it slightly.
  for(int i = 0; i < M-2; i++) {
    for(int j = 0; j < N; j++) {
      h_T[i*N + j] = 1 - h_Z[i] + 0.01*sin(PI*h_Z[i])*cos((PI/XF)*h_X[j]);
      // For debugging purposes
      //h_T[i*N + j] = i + 1;
    }
  }

  // Define the arrays necessary to calculate the Nusselt number.
  // These will be transferred to the GPU.
  float* h_ztop = (float*)malloc(4*N*sizeof(float));
  float* h_zbot = (float*)malloc(4*N*sizeof(float));


  for(int i = 0; i < 4*N; i++) {
    h_ztop[i] = 0.f;
    h_zbot[i] = 0.f;
  }

  for(int i = 0; i < N; i++ ) {
    // First row of h_ztop
    h_ztop[i] = 1-3*DX;
    // Second row of h_ztop
    h_ztop[i + N] = 1-2*DX;
    // Third row of h_ztop
    h_ztop[i + 2*N] = 1-DX;

    // Second row of h_zbot
    h_zbot[i + N] = h_Z[0];
    // Third row of h_zbot
    h_zbot[i + 2*N] = h_Z[1];
    // Fourth row of h_zbot
    h_zbot[i + 3*N] = h_Z[2];
  }

  // The bottom row of ztop is 1.f and the top row of zbot is 0.f
  for(int i = 3*N; i < 4*N; i++) {
    h_ztop[i] = 1.f;
  }

  // Shadow T in GPU memory.  Although cublasSetMatrix assumes its input is
  // in column-major,
  cublasSetVector(N*(M-2), sizeof(float), h_T, 1, d_T, 1);

  float* d_omega;
  cudaMalloc((void**)&d_omega, (M-2)*(N-2) * sizeof(float));

  float* d_psi;
  cudaMalloc((void**)&d_psi, (M-2)*N * sizeof(float));

  float* d_dsc;
  cudaMalloc((void**)&d_dsc, (M-2)*(M-2) * sizeof(float));

  float* d_dsr;
  cudaMalloc((void**)&d_dsr, (N-2)*(N-2) * sizeof(float));

  float* d_ei;
  cudaMalloc((void**)&d_ei, (N-2)*(M-2) * sizeof(float));

  float* d_tr;
  cudaMalloc((void**)&d_tr, N*M * sizeof(float));

  float* d_trnu;
  cudaMalloc((void**)&d_trnu, N * sizeof(float));

  float* d_ztop, *d_zbot, *d_nutop, *d_nubot;
  cudaMalloc((void**)&d_ztop, 4*N * sizeof(float));
  cudaMalloc((void**)&d_zbot, 4*N * sizeof(float));
  cudaMalloc((void**)&d_nutop, 4*N * sizeof(float));
  cudaMalloc((void**)&d_nubot, 4*N * sizeof(float));

  // Initialize dsr, dsc, lambda, mu, ei, and trNu and copy them over
  float* h_dsc = (float*)malloc((M-2)*(M-2)*sizeof(float));
  float* h_dsr = (float*)malloc((N-2)*(N-2)*sizeof(float));
  float* h_lambda = (float*)malloc((M-2)*sizeof(float));
  float* h_mu = (float*)malloc((N-2)*sizeof(float));
  float* h_ei = (float*)malloc((M-2)*(N-2)*sizeof(float));
  float* h_tr = (float*)malloc(M*N*sizeof(float));
  float* h_trnu = (float*)malloc(N*sizeof(float));
  // Set the value of h_dsc
  for(int i = 0; i < M-2; i++) {
    for(int j = 0; j < M-2; j++) {
      h_dsc[(M-2)*i + j] = sqrtf(2.f/(M-1.f))*sin((i+1.f)*(j+1.f)*PI/(M-1.f));
      // For debugging purposes.
      //if(i == j) h_dsc[(M-2)*i + j] = 1;
      //else h_dsc[(M-2)*i + j] = 0;
    }
  }

  // Set the value of h_dsr.
  for(int i = 0; i < N-2; i++) {
    for(int j = 0; j < N-2; j++) {
      h_dsr[(N-2)*i + j] = sqrtf(2.f/(N-1.f))*sin((i+1.f)*(j+1.f)*PI/(N-1.f));
      // For debugging purposes.
      //if(i == j) h_dsr[(N-2)*i + j] = 1;
      //else h_dsr[(N-2)*i + j] = 0;
    }
  }

  // Initialize lambda and mu, which are used to compute ei.
  for(int i = 0; i < M-2; i++) {
    h_lambda[i] = 2.f*cos((i + 1.f)*PI/(M - 1.f)) - 2.f;
  }

  for(int i = 0; i < N-2; i++) {
    h_mu[i] = 2.f*cos((i + 1.f)*PI/(N - 1.f)) - 2.f;
  }
  // Compute ei from lambda and mu.
  // The elements of ei are inverted on the last step to replace later
  // divisions by multiplications.
  for(int i = 0; i < M-2; i++) {
    for(int j = 0; j < N-2; j++) {
      h_ei[(N-2)*i + j] = h_lambda[i] + h_mu[j];
      h_ei[(N-2)*i + j] = (h_ei[(N-2)*i + j])*(h_ei[(N-2)*i + j]);
      h_ei[(N-2)*i + j] = 1.f/(h_ei[(N-2)*i + j]);
    }
  }
  // Compute tr
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      h_tr[N*i + j] = DX*DX/4.f;
      if(j>0 && j<(M-1)) h_tr[N*i + j] = DX2/2.f;
      if(i>0 && i<(N-1)) h_tr[N*i + j] = DX2/2.f;
      if(j>0 && j<(M-1) && i>0 && i<(N-1)) h_tr[N*i + j] = DX2;
    }
  }

  // Compute trnu
  for(int i = 1; i < N-1; i++) {
    h_trnu[i] = DX;
  }
  h_trnu[0] = DX/2.f;
  h_trnu[N-1] = DX/2.f;

  // Copy the completed data over.
  cublasSetVector((M-2)*(M-2), sizeof(float), h_dsc, 1, d_dsc, 1);
  cublasSetVector((N-2)*(N-2), sizeof(float), h_dsr, 1, d_dsr, 1);
  cublasSetVector((M-2)*(N-2), sizeof(float), h_ei, 1, d_ei, 1);
  cublasSetVector(M*N, sizeof(float), h_tr, 1, d_tr, 1);
  cublasSetVector(N, sizeof(float), h_trnu, 1, d_trnu, 1);
  cublasSetVector(4*N, sizeof(float), h_ztop, 1, d_ztop, 1);
  cublasSetVector(4*N, sizeof(float), h_ztop, 1, d_nutop, 1);
  cublasSetVector(4*N, sizeof(float), h_zbot, 1, d_zbot, 1);
  cublasSetVector(4*N, sizeof(float), h_zbot, 1, d_nubot, 1);

  float* d_u;
  float* d_v;
  float* d_xrk3;
  float* d_yrk3;
  float* d_zrk3;
  float* d_y;
  float* d_Tbuff;
  float* d_DxT;

  cudaMalloc((void**)&d_u, N*(M-2) * sizeof(float));
  cudaMalloc((void**)&d_v, N*(M-2) * sizeof(float));

  cudaMalloc((void**)&d_xrk3, N*(M-2) * sizeof(float));
                            
  cudaMalloc((void**)&d_yrk3, N*(M-2) * sizeof(float));
                            
  cudaMalloc((void**)&d_zrk3, N*(M-2) * sizeof(float));
  cudaMalloc((void**)&d_y, N*(M-2) * sizeof(float));

  cudaMalloc((void**)&d_Tbuff, N*(M-2) * sizeof(float));

  cudaMalloc((void**)&d_DxT, N*(M-2) * sizeof(float));

  // use d_T to define d_psi, d_u, d_v, and d_y = 0.
  float alpha = -1.f;
  cublasScopy(h, N*(M-2), d_T, 1, d_psi, 1);
  cublasSaxpy(h, N*(M-2), &alpha, d_psi, 1, d_psi, 1);
  cublasScopy(h, N*(M-2), d_psi, 1, d_u, 1);
  cublasScopy(h, N*(M-2), d_psi, 1, d_v, 1);
  cublasScopy(h, N*(M-2), d_psi, 1, d_y, 1);

  //initialize all intermediate matrices to T;
  cublasScopy(h, N*(M-2), d_T, 1, d_xrk3, 1);
  cublasScopy(h, N*(M-2), d_T, 1, d_yrk3, 1);
  cublasScopy(h, N*(M-2), d_T, 1, d_zrk3, 1);
  cublasScopy(h, N*(M-2), d_T, 1, d_Tbuff, 1);
  cublasScopy(h, N*(M-2), d_T, 1, d_DxT, 1);


  // DEBUG
  // temporary buffer for use in d_*rk3 stuff
  float* d_temp;
  cudaMalloc((void**)&d_temp, N*(M-2) * sizeof(float));
  cublasScopy(h, N*(M-2), d_T, 1, d_temp, 1);

  printf("Begin computation: \n");

  //=============================================================================
  //                Computation
  //=============================================================================

  // Begin timestep computation.
  float frames = 0.f;
  int tstep = 0;

  // Variable to store timing information
  auto start = std::chrono::steady_clock::now();

  for(int c = STARTSTEP; c < ENDSTEP; c++) {
    // Use SHORTG macro to call g succinctly.
    // x = g(T,0)
    // z = g(T + (dt/3)*x, 0)
    // z = g(T + (2*dt/3)*z, 1)
    // T = T + (dt/4)*(x + 3z)
    // Store the first part of RK3 in d_xrk3
    SHORTG(d_T, 1, d_xrk3, c);

    // add (dt/3)*d_xrk3 to T, store the result in T temporarily.
    //    cublasSaxpy(h, N*(M-2), (dt/3.f), d_xrk3, 1, d_T, 1);
    //DEBUG
    cublasScopy(h, N*(M-2), d_T, 1, d_temp,  1);
    alpha = dt/3.f;
    cublasSaxpy(h, N*(M-2), &alpha, d_xrk3, 1, d_temp, 1);
    // Compute d_yrk3 = g(T + (dt/3)*d_xrk3, 0) by using the updated T.
    //    SHORTG(d_T, 0, d_yrk3);

    //DEBUG
    SHORTG(d_temp, 0, d_yrk3, 0);

    // return d_T to its original state by subtracting (dt/3)*x
    //    cublasSaxpy(h, N*(M-2), (-(dt/3.f)), d_xrk3, 1, d_T, 1);
    // Add (2*dt/3)*d_yrk3 to T, store the result in T temporarily.
    //    cublasSaxpy(h, N*(M-2), (2.f*(dt/3)), d_yrk3, 1, d_T, 1);
    //DEBUG
    cublasScopy(h, N*(M-2), d_T, 1, d_temp, 1);
    alpha = 2.f*(dt/3.f);
    cublasSaxpy(h, N*(M-2), &alpha, d_yrk3, 1, d_temp, 1);
    // Compute d_zrk3 = g(T + (2*dt/3)*d_yrk3) by using the updated T.
    //    SHORTG(d_T, 1, d_zrk3);

    //DEBUG
    SHORTG(d_temp, 0, d_zrk3, 0);

    // return d_T to its original state by subtracting (2*dt/3)*d_yrk3
    //    cublasSaxpy(h, N*(M-2), (-(2.f*(dt/3))), d_yrk3, 1, d_T, 1);
    // T+= (dt/4)*(x + 3z)
    // Add (dt/4)*d_xrk3 to d_T
    //    cublasSaxpy(h, N*(M-2), (dt/4.f), d_xrk3, 1, d_T, 1);
    // Add 3*(dt/4)*d_zrk3 to d_T
    //    cublasSaxpy(h, N*(M-2), (3.f*(dt/4.f)), d_zrk3, 1, d_T, 1);
    //DEBUG
    alpha = dt/4.f;
    cublasSaxpy(h, N*(M-2), &alpha, d_xrk3, 1, d_T, 1);
    alpha = 3.f * alpha;
    cublasSaxpy(h, N*(M-2), &alpha, d_zrk3, 1, d_T, 1);

    // update the value of dt (in host) from d_dt (in device)
    cublasGetVector(1, sizeof(float), d_dt, 1, &h_dt, 1);
    dt = h_dt;
    frames += dt;

    // keeps track of several Nu samples.
    if(frames > FRAMESIZE) {
      tstep++;
      // Calculate the nusselt number throughout the array and save
      float nunum = NusseltCompute(h, d_T, d_nutop, d_ztop, d_zbot, d_nubot, d_trnu);
      printf("Nusselt number: %.1f\n", nunum);
      frames = 0.f;
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  auto time_sec = time * 1e-9f;
  printf("Total compute time: %f (s)\n", time_sec);
  printf("Average compute time per step: %f (s)\n", time_sec / ENDSTEP);

  free(h_X);
  free(h_Z);
  free(h_T);

  free(h_dsr);
  free(h_dsc);
  free(h_lambda);
  free(h_mu);
  free(h_ei);
  free(h_tr);
  free(h_trnu);
  free(h_ztop);
  free(h_zbot);

  cudaFree(d_temp);
  cudaFree(d_T);
  cudaFree(d_dt);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_xrk3);
  cudaFree(d_yrk3);
  cudaFree(d_zrk3);
  cudaFree(d_y);
  cudaFree(d_Tbuff);
  cudaFree(d_DxT);
  cudaFree(d_omega);
  cudaFree(d_psi);
  cudaFree(d_dsc);
  cudaFree(d_dsr);
  cudaFree(d_ei);
  cudaFree(d_tr);
  cudaFree(d_trnu);
  cudaFree(d_ztop);
  cudaFree(d_zbot);
  cudaFree(d_nutop);
  cudaFree(d_nubot);

  cublasDestroy(h);
  printf("Done.\n");
  return 0;
}
