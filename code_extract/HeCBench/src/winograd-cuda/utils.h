//Scott Grauer-Gray (sgrauerg@gmail.com)
//Functions used across hmpp codes

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

// Can switch DATA_TYPE between float and double
#define DATA_TYPE float

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define MAX_SOURCE_SIZE (0x100000)

// Problem size 
#ifndef MAP_SIZE 
#define MAP_SIZE 1024
#endif

// Thread block dimensions
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8


//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

void WinogradConv2D_2x2_omp(DATA_TYPE* input, DATA_TYPE* output, DATA_TYPE* transformed_filter, size_t* cpu_global_size);
void WinogradConv2D_2x2_filter_transformation(DATA_TYPE* transformed_filter);
void WinogradConv2D_2x2(DATA_TYPE* input, DATA_TYPE* output, DATA_TYPE* transformed_filter);
bool compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu);
double rtclock();
float absVal(float a);
float percentDiff(double val1, double val2);


#endif //UTILS_H
