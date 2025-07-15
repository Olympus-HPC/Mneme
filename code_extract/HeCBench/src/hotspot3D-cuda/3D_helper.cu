#include "3D_helper.h"

#define STR_SIZE 256

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void fatal(const char *s)
{
  fprintf(stderr, "Error: %s\n", s);
}

void readinput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {

  int i,j,k;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 )
    fatal( "The file was not opened" );

  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
      for (k=0; k <= layers-1; k++)
      {
        if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
        if (feof(fp))
          fatal("not enough lines in file");
        //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
        if ((sscanf(str, "%f", &val) != 1))
          fatal("invalid file format");
        vect[i*grid_cols+j+k*grid_rows*grid_cols] = val;
      }
  fclose(fp);  
}


void writeoutput(float *vect, int grid_rows, int grid_cols, int layers, char *file) {
  int i,j,k, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
    printf( "The file was not opened\n" );

  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
      for (k=0; k < layers; k++)
      {
        sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j+k*grid_rows*grid_cols]);
        fputs(str,fp);
        index++;
      }

  fclose(fp);  
}

void computeTempCPU(float *pIn, float* tIn, float *tOut, 
    int nx, int ny, int nz, float Cap, 
    float Rx, float Ry, float Rz, 
    float dt, float amb_temp, int numiter) 
{   float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce = cw =stepDivCap/ Rx;
  cn = cs =stepDivCap/ Ry;
  ct = cb =stepDivCap/ Rz;

  cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int c,w,e,n,s,b,t;
  int x,y,z;
  int i = 0;
  do{
    for(z = 0; z < nz; z++)
      for(y = 0; y < ny; y++)
        for(x = 0; x < nx; x++)
        {
          c = x + y * nx + z * nx * ny;

          w = (x == 0) ? c : c - 1;
          e = (x == nx - 1) ? c : c + 1;
          n = (y == 0) ? c : c - nx;
          s = (y == ny - 1) ? c : c + nx;
          b = (z == 0) ? c : c - nx * ny;
          t = (z == nz - 1) ? c : c + nx * ny;


          tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw +
                    tIn[t]*ct + tIn[b]*cb + (dt/Cap) * pIn[c] + ct*amb_temp;
        }
    float *temp = tIn;
    tIn = tOut;
    tOut = temp; 
    i++;
  }
  while(i < numiter);
}

float accuracy(float *arr1, float *arr2, int len)
{
  float err = 0.0; 
  int i;
  for(i = 0; i < len; i++)
  {
    err += (arr1[i]-arr2[i]) * (arr1[i]-arr2[i]);
  }

  return (float)sqrt(err/len);
}
