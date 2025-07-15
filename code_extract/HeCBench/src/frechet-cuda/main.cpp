/*
 * Compute the discrete Frechet distance between two curves specified by
 * discrete ordered points in n-dimensional space.
 *
 * Based on `DiscreteFrechetDist` by Zachary Danziger,
 * http://www.mathworks.com/matlabcentral/fileexchange/ \
 * 31922-discrete-frechet-distance
 *
 * This implementation omits the computation of the coupling sequence. Use
 * this program if you only want to get the DFD, and get it fast.
 *
 * Implements algorithm from
 * [1] T. Eiter and H. Mannila. Computing discrete Frechet distance.
 *     Technical report 94/64, Christian Doppler Laboratory
 *
 * Copyright (c) 2016, Mikhail Pak
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* sqrt, fabs, fmin, fmax */
#include <random>

#define n_d 10000 /* `n_d` : Number of dimensions */

double norm1(int i, int j, const double *c1, const double *c2)
{
  double dist, diff; /* Temp variables for simpler computations */
  int k; /* Index for iterating over dimensions */

  /* Initialise distance */
  dist = 0.0;

  for (k = 0; k < n_d; k++)
  {
    /*
     * Compute the distance between the k-th component of the i-th point
     * of the 1st curve and the k-th component of the j-th point of the
     * 2nd curve.
     *
     * Notice the 1-offset added for better readability (as in [1]).
     */
    diff = *(c1 + (i - 1)*n_d + k) - *(c2 + (j - 1)*n_d + k);
    /* Increment the accumulator variable with the absolute distance */
    dist += fabs(diff);
  }

  return dist;
}

double recursive_norm1(int i, int j, int n_2, double *ca,
                       const double *c1, const double *c2)
{
  /*
   * Target the shortcut to the (i, j)-th entry of the matrix `ca`
   *
   * Once again, notice the 1-offset.
   */
  double *ca_ij = ca + (i - 1)*n_2 + (j - 1);

  /* This implements the algorithm from [1] */
  if (*ca_ij > -1.0) 
  {
    return *ca_ij;
  }
  else if ((i == 1) && (j == 1))
  {
    *ca_ij = norm1(1, 1, c1, c2);
  }
  else if ((i > 1) && (j == 1))
  {
    *ca_ij = fmax(recursive_norm1(i - 1, 1, n_2, ca, c1, c2), norm1(i, 1, c1, c2));
  }
  else if ((i == 1) && (j > 1))
  {
    *ca_ij = fmax(recursive_norm1(1, j - 1, n_2, ca, c1, c2), norm1(1, j, c1, c2));
  }
  else if ((i > 1) && (j > 1))
  {
    *ca_ij = fmax(
        fmin(fmin(
            recursive_norm1(i - 1, j    , n_2, ca, c1, c2),
            recursive_norm1(i - 1, j - 1, n_2, ca, c1, c2)),
            recursive_norm1(i,     j - 1, n_2, ca, c1, c2)),
        norm1(i, j, c1, c2));
  }
  else
  {
    *ca_ij = INFINITY;
  }

  return *ca_ij;
}

void distance_norm1 (
  int n_1, int n_2,
  double *__restrict__ ca,
  const double *__restrict__ c1,
  const double *__restrict__ c2)
{
  for (int i = 1; i <= n_1; i++)
    for (int j = 1; j <= n_2; j++)
      recursive_norm1(i, j, n_2, ca, c1, c2);
}

void discrete_frechet_distance(const int s, const int n_1, const int n_2, const int iter)
{
  double *ca, *c1, *c2;
  int k; /* Index for initialisation of `ca`*/

  int ca_size = n_1*n_2*sizeof(double);
  int c1_size = n_1*n_d*sizeof(double);
  int c2_size = n_2*n_d*sizeof(double);

  /* `ca` : Search array (refer to [1], Table 1, matrix `ca`) */
  ca = (double *) malloc (ca_size);

  /* `c1` and `c2` : Arrays with the 1st and 2nd curve's points respectively */
  c1 = (double *) malloc (c1_size);
  c2 = (double *) malloc (c2_size);

  /* Initialise it with -1.0 */
  for (k = 0; k < n_1*n_2; k++)
  {
    ca[k] = -1.0;
  }

  std::mt19937 gen(19937);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  for (k = 0; k < n_1 * n_d; k++)
  {
    c1[k] = dis(gen);
  }

  for (k = 0; k < n_2 * n_d; k++)
  {
    c2[k] = dis(gen);
  }

  distance_norm1(n_1, n_2, ca, c1, c2);

  double checkSum = 0;
  for (k = 0; k < n_1 * n_2; k++)
    checkSum += ca[k];
  printf("checkSum: %lf\n", checkSum);

  /* Free memory */
  free(ca);
  free(c1);
  free(c2);
}

int main(int argc, char* argv[])
{
  /* `n_1` and `n_2` : Number of points of the 1st and 2nd curves */
  const int n_1 = atoi(argv[1]);
  const int n_2 = atoi(argv[2]);
  const int iter = atoi(argv[3]);

  for (int i = 0; i < 3; i++)
    discrete_frechet_distance(i, n_1, n_2, iter);

  return 0;
}

