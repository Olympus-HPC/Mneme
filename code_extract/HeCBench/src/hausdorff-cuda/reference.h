#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif

#if defined(_OPENMP)
typedef struct __attribute__((__aligned__(8)))
{
  float x,y;
}
float2;
#endif

inline float hd (const float2 ap, const float2 bp);

int cmpfunc (const void * a, const void * b) {
   return ( *(float*)a - *(float*)b ) > 0.f ? 1 : 0;
}

float computeDirDistance(const float2 Apoints[],
                         const float2 Bpoints[],
                         int numA, int numB)
{
  float *disA = (float*) malloc (sizeof(float) * numA);

  for (int i = 0; i < numA; i++)
  {
    float d = std::numeric_limits<float>::max();
    for (int j = 0; j < numB; j++)
    {
      float t = hd(Apoints[i], Bpoints[j]);
      d = std::min(t, d);
    }
    disA[i] = d;
  }
  qsort(disA, numA, sizeof(float), cmpfunc);
  float dis = disA[numA - 1];

  free(disA);
  return dis;
}

float hausdorff_distance(const float2 Apoints[],
                         const float2 Bpoints[],
                         int numA, int numB)
{
  float hAB = computeDirDistance(Apoints, Bpoints, numA, numB);
  float hBA = computeDirDistance(Bpoints, Apoints, numB, numA);
  return std::max(hAB, hBA);  
}
