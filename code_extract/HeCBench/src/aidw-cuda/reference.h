#define a1 1.5f
#define a2 2.f
#define a3 2.5f
#define a4 3.f
#define a5 3.5f
#define R_min 0.f
#define R_max 2.f

// thread block size
#define BLOCK_SIZE  256

// accuracy of 1e-1f is not met with the fast math option across compilers
#define EPS 1

void reference(
  const float *__restrict dx, 
  const float *__restrict dy,
  const float *__restrict dz,
  const int dnum,   // Data points
  const float *__restrict ix,
  const float *__restrict iy,
        float *__restrict iz,
  const int inum,   // Interplated points
  const float area, // Area of planar region
  const float *__restrict avg_dist) 

{
  #pragma omp parallel for
  for (int tid = 0; tid < inum; tid++) {
    float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 0.f;

    float r_obs = avg_dist[tid];                // The observed average nearest neighbor distance
    float r_exp = 1.f / (2.f * sqrtf(dnum / area)); // The expected nearest neighbor distance for a random pattern
    float R_S0 = r_obs / r_exp;                 // The nearest neighbor statistic

    // Normalize the R(S0) measure such that it is bounded by 0 and 1 by a fuzzy membership function 
    float u_R = 0.f;
    if(R_S0 >= R_min) u_R = 0.5f-0.5f * cosf(3.1415926f / R_max * (R_S0 - R_min));
    if(R_S0 >= R_max) u_R = 1.f;

    // Determine the appropriate distance-decay parameter alpha by a triangular membership function
    // Adaptive power parameter: a (alpha)
    if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
    if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
    if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
    if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
    if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
    if(u_R>0.9f && u_R<=1.f)  alpha = a5;
    alpha *= 0.5f; // Half of the power

    // Weighted average
    for(int j = 0; j < dnum; j++) {
      dist = (ix[tid] - dx[j]) * (ix[tid] - dx[j]) + (iy[tid] - dy[j]) * (iy[tid] - dy[j]) ;
      t = 1.f / powf(dist, alpha);  sum += t;  z += dz[j] * t;
    }
    iz[tid] = z / sum;
  }
}


bool verify(float *gold, float *test, const int len, const float eps) {
  for(int i = 0; i < len; i++) {
    if (fabsf(gold[i] - test[i]) > eps) {
      printf("%d %f %f\n", i, gold[i], test[i]);
      return false;
    }
  }
  return true;
}
