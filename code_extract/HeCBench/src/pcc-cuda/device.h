#ifndef DEVICE_H
#define DEVICE_H

// adjust the values for vendors' GPUs
#define THREAD_BLOCK_SIZE 1024
#define FREE_MEMORY       1024UL*1024*1024*4

int CorMat_singlePass(float* , float * , int , int);
int CorMat_multiPass(float* , float * , int , int);
size_t remaining_B(int , size_t );
void preprocessing(float * , int , int );

#endif
