#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include "device.h"

int CorMat_singlePass(float* , float * , int , int);
int CorMat_multiPass(float* , float * , int , int);

size_t remaining_B(int N, size_t available_memory)
{
  size_t x=available_memory;
  size_t temp=N;
  temp*=2;
  x/=temp;
  return x;
}

void preprocessing(float * data_t, int N,int L)
{
  for (int i = 0; i < N; i++)
  {
    float * row = data_t + i * L;
    double sum1 = 0, sum2 = 0;
    for (int l = 0; l < L; l++)
    {
      sum1 += row[l];
    }
    sum1 /= L;
    for (int l = 0; l < L; l++)
    {
      sum2 += (row[l] - sum1) * (row[l] - sum1);
    }
    sum2 = sqrt(sum2);
    for (int l = 0; l < L; l++)
    {
      if(sum2!=0)
        row[l] = (row[l] - sum1) / sum2;
      else
        if(sum2==0)
          row[l]=0;
    }
  }
}


int main(int argc, char *argv[])
{
  if (argc != 3) {
     std::cout<<"Usage: " << argv[0] << " <Number of voxels> <Length of time series>\n";
     return 1;
  }

  int N = atoi(argv[1]);
  int L = atoi(argv[2]);
  std::cout<<"Number of voxels: "<<N<<"  "<<"Length of time series: "<<L<<"\n\n";

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> uniform_distr (-6.f, 6.f);

  int k = 0, l = 0;
  float * data = new float [L * N];
  for (  k = 0; k < N; k++)
    for ( l = 0; l < L; l++)
      data[k*L+l] = uniform_distr(g);

  size_t M11 = (N-1);
  M11 *= N;
  M11 /= 2;

  float * upper_tri = new float [M11];

  for(size_t indii=0;indii<M11;indii++)
    upper_tri[indii]=0;

  std::cout<<"\nComputing correlations ...\n";

  const size_t free_memory = FREE_MEMORY;

  size_t app_memory = sizeof(float) * ((size_t)N * L + (size_t)N * N + M11);

  auto start = std::chrono::steady_clock::now();

  if (app_memory < free_memory) {
    CorMat_singlePass(upper_tri, data, N, L);
  }
  else {
    CorMat_multiPass(upper_tri, data, N, L);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout<<"\nRunning time for computing correlations: \n"<< (time * 1e-9f) << " (s)\n";

  double checksum = 0;
  for(size_t tab =0;tab<M11;tab++) {
    checksum += upper_tri[tab];
  }
  std::cout<<"Checksum: " << checksum << "\n";

  if (N < 100 && L < 100) {
    std::cout<<"\nWriting correlation values into the text file ... \n";
    std::ofstream correlations_print;
    correlations_print.open("corrs.txt");
    for(size_t tab =0;tab<M11;tab++) {
      correlations_print << upper_tri[tab] << '\n';
    }
    correlations_print.close();
    std::cout<<"\nCorrelations are stored into the text file corrs.txt \n";
  }

  delete [] upper_tri;
  delete [] data;

  return 0;
}
