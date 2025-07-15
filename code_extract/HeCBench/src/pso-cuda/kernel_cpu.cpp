#include "kernel.h"

float host_fitness_function(float x[])
{
  float res = 0.f;
  float y1 = F(x[0]);
  float yn = F(x[DIM-1]);

  res += powf(sinf(phi*y1), 2.f) + powf(yn-1, 2.f);

  for(int i = 0; i < DIM-1; i++)
  {
    float y = F(x[i]);
    float yp = F(x[i+1]);
    res += powf(y-1.f, 2.f) * (1.f + 10.f * powf(sinf(phi*yp), 2.f));
  }

  return res;
}

float getRandom(float low,float high)
{
  return low+float(((high-low)+1.f)*rand()/((float)RAND_MAX+1.f));
}

float getRandomClamped(int seed)
{
  srand(seed);
  return (float)rand()/(float)RAND_MAX;
}

void pso(int p,int r,float *positions,float *velocities,float *pBests,float *gBest)
{
  float tempParticle1[DIM];
  float tempParticle2[DIM];

  for(int iter=0;iter<r;iter++)
  {
    float rp=getRandomClamped(iter);
    float rg=getRandomClamped(r-iter);

    for(int i=0;i<p*DIM;i++)
    {
      velocities[i]=OMEGA*velocities[i]+
                    c1*rp*(pBests[i]-positions[i])+
                    c2*rg*(gBest[i%DIM]-positions[i]);

      positions[i]+=velocities[i];
    }

    for(int i=0;i<p*DIM;i+=DIM)
    {
      for(int j=0;j<DIM;j++)
      {
        tempParticle1[j]=positions[i+j];
        tempParticle2[j]=pBests[i+j];
      }
      if(host_fitness_function(tempParticle1)<host_fitness_function(tempParticle2))
      {
        for(int j=0;j<DIM;j++)
          pBests[i+j]=tempParticle1[j];

        if(host_fitness_function(tempParticle1)< 130.f) //host_fitness_function(gBest))
        {
          for(int j=0;j<DIM;j++) {
            gBest[j] += tempParticle1[j];
          }
        }
      }
    }
  }
}
