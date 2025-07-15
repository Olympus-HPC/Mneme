#define one_over_theta0  (5.0/(2.0*pow(M_PI,4.0)))
#define kappa (pow(M_PI,2.0)/3.0 - 2 * 1.2020569031595942)

/*
 there are some limitation on L_max
 due to GSL's factorial N! is only defined in GSL for N<=170
 thus L_max < 85 
 */
const int L_max=64;

__device__ double B(int l)
{
  return 2.0*(14.0*l*l + 7.0*l - 2.0)/(4.0*l-1.0)/(4.0*l+3.0);
}

__device__ double U(int l)
{
  return -(2.0*l-1)*(2.0*l+1)*(2.0*l+2)/(4.0*l+3)/(4.0*l+5);
}

__device__ double C(int l)
{
  return (2.0*l-1)*2.0*l*(2.0*l+2)/(4.0*l-3.0)/(4.0*l-1.0);
}

__device__ double alpha(int l)
{
  if(l<0) return 0.0;
  if(l==0) return 1.0;
  else 
    return double_fact_table[2*l-1] / fact_table[l];
}

__device__ double Omega(int l, int m, int n)
{
  return alpha(m-n+l) * alpha(m+n-l) * alpha(n-m+l)/alpha(m+n+l) * (4*l+1) / (2.0*(n+m+l)+1) ;
}

__device__ double Sum_Omega(int l, const double c[])
{
  double sum = 0.0;
  for(int m=1; m<L_max; m++)
    for(int n=1; n<L_max; n++)
      if(abs(m-n)<l+1) sum += Omega(l,m,n) * c[m]*c[n];
  return sum;
}

__device__ double Sum_NL(int l, const double c[])
{
  double sum = 0.0;
  for(int n=1; n<L_max; n++)
    sum += pow(c[n],2.0)/(4.0*n+1.0);
  return sum * (2*l-1)*(l+1)*c[l]/3.0;
}

__global__ void RHS_f (double t, const double *c, double *RHS)
{
  int l = threadIdx.x;

  double T = c[0];

  if (l == 0) {
    RHS[0] = - T/3.0/t *(1.0+0.1*c[1]);
  } else if (l < L_max) {
    double B_bar = B(l) - 4.0/3.0;
    double LHS_119;
    if (l > 1) 
      LHS_119 = 1.0/t * ( U(l) * c[l+1] + (B_bar - 2.0/15.0 * c[1]) + C(l) * c[l-1]);
    else 
      LHS_119 = 1.0/t * ( U(1) * c[2] + (B_bar - 2.0/15.0 * c[1]) + C(1));

    double Sum1 = Sum_Omega(l, c);
    double Sum2 = Sum_NL(l, c);

    double RHS_119 = - T*one_over_theta0 *(
        (kappa + M_PI*M_PI*l*(2*l+1)/3.0)*c[l] + 
         kappa*Sum1 + kappa*Sum2);

    RHS[l] = -LHS_119 + RHS_119;
  }
}
