#ifndef LINEAR_H__
#define LINEAR_H__

#ifdef __NVCC__
#include <cuda.h>
#else
#include <hip/hip_runtime.h>
#endif

#define RESULT_FILENAME "assets/_results.txt"
#define TEMP_FILENAME "assets/temperature.txt"

#define TEMP_SIZE 96453
#define TEMP_WORKGROUP_SIZE 63
#define TEMP_WORKGROUP_NBR (TEMP_SIZE / TEMP_WORKGROUP_SIZE)

#define LOG_DATASET() \
  for (int i = 0; i < DATASET_SIZE; i++) \
    printf("(%f, %f)\n", dataset[i].x, dataset[i].y);

#define LOG_DATA_T(var) printf("data_t:\n\tx: %f\n\ty: %f\n", var.x, var.y);
#define LOG_SUM_T(var) printf("sum_t:\n\tsumx: %f\n\tsumy: %f\n\tsumxy: %f\n\tsumxsq: %f\n", var.x, var.y, var.z, var.w);
#define LOG_RESULT_T(var) printf("result_t:\n\ta0: %f\n\ta1: %f\n\ttime: %f\n", var.a0, var.a1, var.time);
#define LOG_RSQUARED_T(var) printf("rsquared_t:\n\tactual: %f\n\testimated: %f\n", var.x, var.y)

/* a0 # a1 # time # rsquared */
#define WRITE_RESULT(file, result) \
  fprintf(file, "%.3f   %.3f   %3.f   %d\n", \
    result.a0, \
    result.a1, \
    result.ktime * 1e-6, \
    result.rsquared);

#define PRINT_RESULT(title, result) \
  printf("\t%s\n\t--------\n\t| Equation: y = %.3fx + %.3f\n\n", \
    title, \
    result.a1, \
    result.a0);

typedef struct {
    int repeat;
 char * filename;
 size_t size;
 size_t wg_size;
 size_t wg_count;
} linear_param_t;

typedef float2 data_t;

typedef float4 sum_t;

typedef struct {
  float a0;
  float a1;
  int rsquared;
  double ktime;  // total kernel execution time
} result_t;

typedef struct {
  result_t iterative;
  result_t parallelized;
} results_t;

typedef float2 rsquared_t;

void parallelized_regression(linear_param_t *, data_t *, result_t *);
void iterative_regression(linear_param_t *, data_t *, result_t *);

#endif
