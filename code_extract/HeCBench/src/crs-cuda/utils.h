#ifndef __UTILS
#define __UTILS

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

size_t align_value(size_t valueToAlign, size_t alignMask);

void generateRandomValue(char *data, size_t size);
typedef void (*coding_func)(int k, int index,
    char *dataPtr, char *codeDevPtr,
    const unsigned int *bitMatrixPtr,
    int threadDimX,int blockDimX,
    int workSizePerGridInLong);

double elapsed_time_in_ms(struct timeval startTime, struct timeval endTime);
 

#define talloc(type, num) (type *)malloc(sizeof(type) * (num))

#define MIN_K 1
#define MAX_K 4

#define MIN_M 1
#define MAX_M 4

#define MIN_W 4
#define MAX_W 8

#define MAX_K_MULIPLY_M (MAX_K * MAX_M)

#define WARM_UP_SIZE 4

#define MAX_THREAD_NUM 128
#define THREADS_PER_BLOCK 128
#define WARP_SIZE 32

#endif

