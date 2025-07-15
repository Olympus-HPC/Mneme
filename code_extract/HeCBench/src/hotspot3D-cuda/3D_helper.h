#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

void  fatal(const char* s);
void  readinput(float* v, int r, int c,int l,char*);
void  writeoutput(float* v,int r,int c,int l,char*);
long long get_time(); 
float accuracy(float* arr1, float* arr2, int len);
void computeTempCPU(float* pIn, float *tIn, float *tOut, 
               int nx, int ny, int nz, float Cap,
               float Rx, float Ry, float Rz, 
               float dt, float amb_temp, int numiter); 
