#ifndef _GAUSSIANELIM
#define _GAUSSIANELIM

#include <iostream>
#include <vector>
#include <float.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>


void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,
                     int *q, int *t, int *size);
                     
void InitPerRun(int size,float *m);
void ForwardSub(float *a, float *b, float *m, int size,int timing);
void BackSub(float *a, float *b, float *finalVec, int size);
void Fan1(float *m, float *a, int Size, int t);
void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
//void Fan3(float *m, float *b, int Size, int t);
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol);
void InitAry(FILE *fp, float *ary, int ary_size);
void PrintMat(float *ary, int size, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
#endif
