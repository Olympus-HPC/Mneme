/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Tridiagonal solvers.
 * Main host code.
 *
 * This sample implements several methods to solve a bunch of small tridiagonal matrices:
 *  PCR    - parallel cyclic reduction O(N log N)
 *  CR    - original cyclic reduction O(N)
 *  Sweep  - serial one-thread-per-system gauss elimination O(N)
 *
 * Original testrig code: UC Davis, Yao Zhang & John Owens
 * Reference paper for the cyclic reduction methods on the GPU:  
 *   Yao Zhang, Jonathan Cohen, and John D. Owens. Fast Tridiagonal Solvers on the GPU. 
 *   In Proceedings of the 15th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP 2010), January 2010.
 * 
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "shrUtils.h"
#include "file_read_write.h"
#include "test_gen_result_check.h"
#include "cpu_solvers.h"

bool             useLmem = false;  // select sweep_small_systems_local_kernel
bool             useVec4 = false;  // select sweep_small_systems_global_vec4_kernel
int              SWEEP_BLOCK_SIZE = 256;

// available solvers
#include "pcr_small_systems.h"
#include "cyclic_small_systems.h"
#include "sweep_small_systems.h"

////////////////////////////////////////////////////////////////////////////////
// Solve <num_systems> of <system_size> using <devCount> devices
////////////////////////////////////////////////////////////////////////////////
void run(int system_size, int num_systems)
{
  double time_spent_gpu[3];
  double time_spent_cpu[1];

  const unsigned int mem_size = sizeof(float) * num_systems * system_size;

  // allocate host arrays
  float *a = (float*)malloc(mem_size);
  float *b = (float*)malloc(mem_size);
  float *c = (float*)malloc(mem_size);
  float *d = (float*)malloc(mem_size);
  float *x1 = (float*)malloc(mem_size);
  float *x2 = (float*)malloc(mem_size);  // result

  // fill host arrays with data
  for (int i = 0; i < num_systems; i++)
    test_gen_cyclic(&a[i * system_size], &b[i * system_size], &c[i * system_size], 
                    &d[i * system_size], &x1[i * system_size], system_size, 0);

  shrLog("  Num_systems = %d, system_size = %d\n", num_systems, system_size);

  // run CPU serial solver
  time_spent_cpu[0] = serial_small_systems(a, b, c, d, x2, system_size, num_systems);

  // Log info
  shrLog("\n----- CPU  solvers -----\n");
  shrLog("  CPU Time =    %.5f s\n", time_spent_cpu[0]);
  shrLog("  Throughput =  %.4f systems/sec\n", (float)num_systems /(time_spent_cpu[0]*1000.0));

  // run GPU solvers
  shrLog("\n----- optimized GPU solvers -----\n\n");

  // pcr (base and optimized)
  time_spent_gpu[0] = pcr_small_systems(a, b, c, d, x1, system_size, num_systems, 0);
  shrLogEx(LOGBOTH | MASTER, 0, "Tridiagonal-pcrsmall-base, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
      (1.0e-3 * (double)num_systems / time_spent_gpu[0]), time_spent_gpu[0], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[0] = pcr_small_systems(a, b, c, d, x1, system_size, num_systems, 1);
  shrLogEx(LOGBOTH | MASTER, 0, "Tridiagonal-pcrsmall-optimized, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
      (1.0e-3 * (double)num_systems / time_spent_gpu[0]), time_spent_gpu[0], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  // cyclic (base and optimized)
  time_spent_gpu[1] = cyclic_small_systems(a, b, c, d, x1, system_size, num_systems, 0);
  shrLogEx(LOGBOTH | MASTER, 0, "Tridiagonal-cyclicsmall-base, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n", 
      (1.0e-3 * (double)num_systems / time_spent_gpu[1]), time_spent_gpu[1], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  time_spent_gpu[1] = cyclic_small_systems(a, b, c, d, x1, system_size, num_systems, 1);
  shrLogEx(LOGBOTH | MASTER, 0, "Tridiagonal-cyclicsmall-optimized, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n", 
      (1.0e-3 * (double)num_systems / time_spent_gpu[1]), time_spent_gpu[1], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

  // sweep 
  // Warning: reorder must be enabled for the global vec4 kernel to produce meaning results
  if (!useVec4) {
    time_spent_gpu[2] = sweep_small_systems(a, b, c, d, x1, system_size, num_systems, false);
    shrLogEx(LOGBOTH | MASTER, 0, "Tridiagonal-sweepsmall-noreorder, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
          (1.0e-3 * (double)num_systems / time_spent_gpu[2]), time_spent_gpu[2], num_systems);
      compare_small_systems(x1, x2, system_size, num_systems);
  }

  time_spent_gpu[2] = sweep_small_systems(a, b, c, d, x1, system_size, num_systems, true);
  shrLogEx(LOGBOTH | MASTER, 0, "Tridiagonal-sweepsmall-reorder, Throughput = %.4f Systems/s, Time = %.5f s, Size = %u Systems\n",
      (1.0e-3 * (double)num_systems / time_spent_gpu[2]), time_spent_gpu[2], num_systems);
  compare_small_systems(x1, x2, system_size, num_systems);

#ifdef OUTPUT_RESULTS
  const char* gpu_data_file = "TriDiagonal_GPU.dat";
  const char* cpu_data_file = "TriDiagonal_CPU.dat";
  const char* gpu_time_file = "TriDiagonal_GPU.timing";
  const char* cpu_time_file = "TriDiagonal_CPU.timing";
  file_write_small_systems(x1, 10, system_size, gpu_data_file);
  file_write_small_systems(x2, 10, system_size, cpu_data_file);
  write_timing_results_1d(time_spent_gpu, 1, gpu_time_file);
  write_timing_results_1d(time_spent_cpu, 1, cpu_time_file);
#endif 

  // free host arrays
  free(a);
  free(b);
  free(c);
  free(d);
  free(x1);
  free(x2);
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char** argv) 
{   
  // set logfile name and start logs
  shrSetLogFileName ("oclTridiagonal.txt");
  shrLog("%s Starting...\n\n", argv[0]); 

  int num_systems = 128 * 128;
  int system_size = 128;

  if(shrCheckCmdLineFlag(argc, (const char**)argv, "num_systems"))
  {
    char* ctaList;
    char* ctaStr;
#ifdef WIN32
    char* next_token;
#endif
    shrGetCmdLineArgumentstr(argc, (const char**)argv, "num_systems", &ctaList);

#ifdef WIN32
    ctaStr = strtok_s (ctaList," ,.-", &next_token);
#else
    ctaStr = strtok (ctaList," ,.-");
#endif

    num_systems = atoi(ctaStr);
  }

// system size must be less than 128. The system size is 128 in sweep kernel
// float a[128]
  if(shrCheckCmdLineFlag(argc, (const char**)argv, "system_size"))
  {
    char* ctaList;
    char* ctaStr;
#ifdef WIN32
    char* next_token;
#endif
    shrGetCmdLineArgumentstr(argc, (const char**)argv, "system_size", &ctaList);

#ifdef WIN32
    ctaStr = strtok_s (ctaList," ,.-", &next_token);
#else
    ctaStr = strtok (ctaList," ,.-");
#endif

    // 128 is the array size in the sweep_small_systems_local_kernel
    // local memory: float a[128];
    system_size = atoi(ctaStr);
    if (system_size > 128) {
      shrLog("system size must be no more than 128\n");
      return -1;
    }
  }

  // check lmem flag
  if (shrCheckCmdLineFlag(argc, (const char**)argv, "lmem"))
    useLmem = true;

  // check vectorization flag
  if (shrCheckCmdLineFlag(argc, (const char**)argv, "vec4"))
    useVec4 = true;

  // CTA size for the sweep
  if (shrCheckCmdLineFlag(argc, (const char**)argv, "sweep-cta"))
  {
    char* ctaList;
    char* ctaStr;
#ifdef WIN32
    char* next_token;
#endif
    shrGetCmdLineArgumentstr(argc, (const char**)argv, "sweep-cta", &ctaList);

#ifdef WIN32
    ctaStr = strtok_s (ctaList," ,.-", &next_token);
#else
    ctaStr = strtok (ctaList," ,.-");
#endif

    SWEEP_BLOCK_SIZE = atoi(ctaStr);
  }
  if (useVec4) shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE / 4);
  else shrLog("Using CTA of size %i for Sweep\n\n", SWEEP_BLOCK_SIZE);

  // run the tests on a GPU
  run(system_size, num_systems);

  return 0;
}
