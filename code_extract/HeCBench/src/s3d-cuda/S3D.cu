#include <cassert>
#include <chrono>
#include <string>
#include <sstream>
#include <cuda.h>
#include "cudacommon.h"
#include "OptionParser.h"
#include "S3D.h"
#include "gr_base.h"
#include "ratt.h"
#include "ratt2.h"
#include "ratx.h"
#include "qssa.h"
#include "qssa2.h"
#include "rdwdot.h"

using namespace std;

// Forward declaration
template <class real>
void RunTest(string testName, OptionParser &op);

// ********************************************************
// Function: toString
//
// Purpose:
//   Simple templated function to convert objects into
//   strings using stringstream
//
// Arguments:
//   t: the object to convert to a string
//
// Returns:  a string representation
//
// Modifications:
//
// ********************************************************
template<class T> inline string toString(const T& t)
{
  stringstream ss;
  ss << t;
  return ss.str();
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
  void
addBenchmarkSpecOptions(OptionParser &op)
{
  ; // No S3D specific options
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the S3D benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: March 13, 2010
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{
  auto t1 = std::chrono::high_resolution_clock::now();
  RunTest<float>("S3D-SP", op);
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time * 1e-9);

  t1 = std::chrono::high_resolution_clock::now();
  RunTest<double>("S3D-DP", op);
  t2 = std::chrono::high_resolution_clock::now();
  total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time * 1e-9);
}

template <class real>
void RunTest(string testName, OptionParser &op)
{
  // Number of grid points (specified in header file)
  int probSizes_SP[4] = { 24, 32, 40, 48};
  int probSizes_DP[4] = { 16, 24, 32, 40};
  int *probSizes = (sizeof(real) == sizeof(double)) ? probSizes_DP : probSizes_SP;
  int sizeClass = op.getOptionInt("size") - 1;
  assert(sizeClass >= 0 && sizeClass < 4);
  sizeClass = probSizes[sizeClass];
  int n = sizeClass * sizeClass * sizeClass;

  // Host variables
  real* host_t;
  real* host_p;
  real* host_y;
  real* host_wdot;
  real* host_molwt;

  // GPU Variables
  real* gpu_t; //Temperatures array
  real* gpu_p; //Pressures array
  real* gpu_y; //Mass fractions
  real* gpu_wdot; //Output variables

  // GPU Intermediate Variables
  real* gpu_rf, *gpu_rb;
  real* gpu_rklow;
  real* gpu_c;
  real* gpu_a;
  real* gpu_eg;
  real* gpu_molwt;

  // Malloc host memory
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_t,        n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_p,        n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_y, Y_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_wdot,WDOT_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMallocHost((void**)&host_molwt,WDOT_SIZE*sizeof(real)));

  // Initialize Test Problem

  // For now these are just 1 for verification
  real rateconv = 1.0;
  real tconv = 1.0;
  real pconv = 1.0;

  // Initialize temp and pressure
  for (int i=0; i<n; i++)
  {
    host_p[i] = 1.0132e6;
    host_t[i] = 1000.0;
  }

  // Init molwt: for now these are just 1 for verification
  for (int i=0; i<WDOT_SIZE; i++)
  {
    host_molwt[i] = 1;
  }

  // Initialize mass fractions
  for (int j=0; j<Y_SIZE; j++)
  {
    for (int i=0; i<n; i++)
    {
      host_y[(j*n)+i]= 0.0;
      if (j==14)
        host_y[(j*n)+i] = 0.064;
      if (j==3)
        host_y[(j*n)+i] = 0.218;
      if (j==21)
        host_y[(j*n)+i] = 0.718;
    }
  }

  // Malloc GPU memory
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_t, n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_p, n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_y, Y_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_wdot, WDOT_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_rf, RF_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_rb, RB_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_rklow, RKLOW_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_c, C_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_a, A_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_eg, EG_SIZE*n*sizeof(real)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gpu_molwt, WDOT_SIZE*sizeof(real)));

  // Get kernel launch config, assuming n is divisible by block size
  dim3 thrds(BLOCK_SIZE,1,1);
  dim3 blks(n / BLOCK_SIZE,1,1);
  dim3 thrds2(BLOCK_SIZE2,1,1);
  dim3 blks2(n / BLOCK_SIZE2,1,1);

  // Download of gpu_t, gpu_p, gpu_y, gpu_molwt
  CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_t, host_t, n*sizeof(real),
        cudaMemcpyHostToDevice, 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_p, host_p, n*sizeof(real),
        cudaMemcpyHostToDevice, 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_y, host_y, Y_SIZE*n*sizeof(real),
        cudaMemcpyHostToDevice, 0));
  CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_molwt,host_molwt,WDOT_SIZE*sizeof(real),
        cudaMemcpyHostToDevice, 0));

  unsigned int passes = op.getOptionInt("passes");

  cudaDeviceSynchronize();
  auto start  = std::chrono::high_resolution_clock::now();

  for (unsigned int i = 0; i < passes; i++)
  {
    ratt_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, tconv);

    rdsmh_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_eg, tconv);

    gr_base <<< dim3(blks2), dim3(thrds2) >>> ( gpu_p, gpu_t, gpu_y,
        gpu_c, tconv, pconv);

    ratt2_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt3_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt4_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt5_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt6_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt7_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt8_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt9_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rf, gpu_rb,
        gpu_eg, tconv);
    ratt10_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_t, gpu_rklow, tconv);

    ratx_kernel <<< dim3(blks), dim3(thrds) >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb,
        gpu_rklow, tconv);
    ratxb_kernel <<< dim3(blks), dim3(thrds) >>> ( gpu_t, gpu_c, gpu_rf, gpu_rb,
        gpu_rklow, tconv);
    ratx2_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_c, gpu_rf, gpu_rb);
    ratx4_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_c, gpu_rf, gpu_rb);

    qssa_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_a);
    qssab_kernel <<< dim3(blks), dim3(thrds) >>> ( gpu_rf, gpu_rb, gpu_a);
    qssa2_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_a);

    rdwdot_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
    rdwdot2_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
    rdwdot3_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
    rdwdot6_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
    rdwdot7_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
    rdwdot8_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
    rdwdot9_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
    rdwdot10_kernel <<< dim3(blks2), dim3(thrds2) >>> ( gpu_rf, gpu_rb, gpu_wdot,
        rateconv, gpu_molwt);
  }

  cudaDeviceSynchronize();
  auto end  = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("\nAverage time of executing s3d kernels: %lf (us)\n", (time * 1e-3) / passes);

  // Copy back result
  CUDA_SAFE_CALL(cudaMemcpy(host_wdot, gpu_wdot,
        WDOT_SIZE * n * sizeof(real), cudaMemcpyDeviceToHost));

  // Free GPU memory
  CUDA_SAFE_CALL(cudaFree(gpu_t));
  CUDA_SAFE_CALL(cudaFree(gpu_p));
  CUDA_SAFE_CALL(cudaFree(gpu_y));
  CUDA_SAFE_CALL(cudaFree(gpu_wdot));
  CUDA_SAFE_CALL(cudaFree(gpu_rf));
  CUDA_SAFE_CALL(cudaFree(gpu_rb));
  CUDA_SAFE_CALL(cudaFree(gpu_c));
  CUDA_SAFE_CALL(cudaFree(gpu_rklow));
  CUDA_SAFE_CALL(cudaFree(gpu_a));
  CUDA_SAFE_CALL(cudaFree(gpu_eg));
  CUDA_SAFE_CALL(cudaFree(gpu_molwt));

  for (int i=0; i<WDOT_SIZE; i++) {
    printf("% 23.16E ", host_wdot[i*n]);
    if (i % 3 == 2)
      printf("\n");
  }
  printf("\n");

  // Free host memory
  CUDA_SAFE_CALL(cudaFreeHost(host_t));
  CUDA_SAFE_CALL(cudaFreeHost(host_p));
  CUDA_SAFE_CALL(cudaFreeHost(host_y));
  CUDA_SAFE_CALL(cudaFreeHost(host_wdot));
  CUDA_SAFE_CALL(cudaFreeHost(host_molwt));
}
