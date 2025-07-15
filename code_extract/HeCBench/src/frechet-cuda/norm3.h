__device__
double norm3(int i, int j, const double *c1, const double *c2)
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
    /* Update the current maximum  */
    dist = fmax(dist, fabs(diff));
  }

  return dist;
}

__device__
double recursive_norm3(int i, int j, int n_2, double *ca,
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
    *ca_ij = norm3(1, 1, c1, c2);
  }
  else if ((i > 1) && (j == 1))
  {
    *ca_ij = fmax(recursive_norm3(i - 1, 1, n_2, ca, c1, c2), norm3(i, 1, c1, c2));
  }
  else if ((i == 1) && (j > 1))
  {
    *ca_ij = fmax(recursive_norm3(1, j - 1, n_2, ca, c1, c2), norm3(1, j, c1, c2));
  }
  else if ((i > 1) && (j > 1))
  {
    *ca_ij = fmax(
        fmin(fmin(
            recursive_norm3(i - 1, j    , n_2, ca, c1, c2),
            recursive_norm3(i - 1, j - 1, n_2, ca, c1, c2)),
            recursive_norm3(i,     j - 1, n_2, ca, c1, c2)),
        norm3(i, j, c1, c2));
  }
  else
  {
    *ca_ij = INFINITY;
  }

  return *ca_ij;
}

__global__ void distance_norm3 (
  int n_1, int n_2,
  double *__restrict__ ca,
  const double *__restrict__ c1,
  const double *__restrict__ c2)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (j >= 1 && j <= n_2 && i >= 1 && i <= n_1)
    recursive_norm3(i, j, n_2, ca, c1, c2);
}
