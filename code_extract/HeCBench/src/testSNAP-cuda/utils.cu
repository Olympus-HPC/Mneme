// ----------------------------------------------------------------------
// Copyright (2019) Sandia Corporation.
// Under the terms of Contract DE-AC04-94AL85000
// with Sandia Corporation, the U.S. Government
// retains certain rights in this software. This
// software is distributed under the Zero Clause
// BSD License
//
// TestSNAP - A prototype for the SNAP force kernel
// Version 0.0.2
// Main changes: Y array trick, memory compaction
//
// Original author: Aidan P. Thompson, athomps@sandia.gov
// http://www.cs.sandia.gov/~athomps, Sandia National Laboratories
//
// ----------------------------------------------------------------------

/* ----------------------------------------------------------------------

   this implementation is based on the method outlined
   in Bartok[1], using formulae from VMK[2].

   for the Clebsch-Gordan coefficients, we
   convert the VMK half-integral labels
   a, b, c, alpha, beta, gamma
   to array offsets j1, j2, j, m1, m2, m
   using the following relations:

   j1 = 2*a
   j2 = 2*b
   j =  2*c

   m1 = alpha+a      2*alpha = 2*m1 - j1
   m2 = beta+b    or 2*beta = 2*m2 - j2
   m =  gamma+c      2*gamma = 2*m - j

   in this way:

   -a <= alpha <= a
   -b <= beta <= b
   -c <= gamma <= c

   becomes:

   0 <= m1 <= j1
   0 <= m2 <= j2
   0 <= m <= j

   and the requirement that
   a+b+c be integral implies that
   j1+j2+j must be even.
   The requirement that:

   gamma = alpha+beta

   becomes:

   2*m - j = 2*m1 - j1 + 2*m2 - j2

   Similarly, for the Wigner U-functions U(J,m,m') we
   convert the half-integral labels J,m,m' to
   array offsets j,ma,mb:

   j = 2*J
   ma = J+m
   mb = J+m'

   so that:

   0 <= j <= 2*Jmax
   0 <= ma, mb <= j.

   For the bispectrum components B(J1,J2,J) we convert to:

   j1 = 2*J1
   j2 = 2*J2
   j = 2*J

   and the requirement:

   |J1-J2| <= J <= J1+J2, for j1+j2+j integral

   becomes:

   |j1-j2| <= j <= j1+j2, for j1+j2+j even integer

   or

   j = |j1-j2|, |j1-j2|+2,...,j1+j2-2,j1+j2

   [1] Albert Bartok-Partay, "Gaussian Approximation..."
   Doctoral Thesis, Cambrindge University, (2009)

   [2] D. A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii,
   "Quantum Theory of Angular Momentum," World Scientific (1988)

------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   factorial n table, size SNA::nmaxfactorial+1
------------------------------------------------------------------------- */

const double nfac_table[] = {
  1,
  1,
  2,
  6,
  24,
  120,
  720,
  5040,
  40320,
  362880,
  3628800,
  39916800,
  479001600,
  6227020800,
  87178291200,
  1307674368000,
  20922789888000,
  355687428096000,
  6.402373705728e+15,
  1.21645100408832e+17,
  2.43290200817664e+18,
  5.10909421717094e+19,
  1.12400072777761e+21,
  2.5852016738885e+22,
  6.20448401733239e+23,
  1.5511210043331e+25,
  4.03291461126606e+26,
  1.08888694504184e+28,
  3.04888344611714e+29,
  8.8417619937397e+30,
  2.65252859812191e+32,
  8.22283865417792e+33,
  2.63130836933694e+35,
  8.68331761881189e+36,
  2.95232799039604e+38,
  1.03331479663861e+40,
  3.71993326789901e+41,
  1.37637530912263e+43,
  5.23022617466601e+44,
  2.03978820811974e+46, // nmaxfactorial = 39
};

/* ----------------------------------------------------------------------
   the function delta given by VMK Eq. 8.2(1)
------------------------------------------------------------------------- */

double deltacg(int j1, int j2, int j)
{
  double sfaccg = factorial((j1 + j2 + j) / 2 + 1);
  return sqrt(factorial((j1 + j2 - j) / 2) * factorial((j1 - j2 + j) / 2) *
              factorial((-j1 + j2 + j) / 2) / sfaccg);
}

/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients using
   the quasi-binomial formula VMK 8.2.1(3)
------------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   pre-compute table of sqrt[p/m2], p, q = 1,twojmax
   the p = 0, q = 0 entries are allocated and skipped for convenience.
   a second table is computed with +/-1 parity factor
------------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

int compute_ncoeff(int twojmax)
{
  int ncount;

  ncount = 0;

  for (int j1 = 0; j1 <= twojmax; j1++)
    for (int j2 = 0; j2 <= j1; j2++)
      for (int j = abs(j1 - j2); j <= MIN(twojmax, j1 + j2); j += 2)
        if (j >= j1)
          ncount++;

  return ncount;
}

/* ---------------------------------------------------------------------- */
__device__
double compute_sfac(double r, double rcut, const int switch_flag)
{
  if (switch_flag == 0)
    return 1.0;
  if (switch_flag == 1) {
    if (r <= rmin0)
      return 1.0;
    else if (r > rcut)
      return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }
  }
  return 0.0;
}

/* ---------------------------------------------------------------------- */
__device__
double compute_dsfac(double r, double rcut, const int switch_flag)
{
  if (switch_flag == 0)
    return 0.0;
  if (switch_flag == 1) {
    if (r <= rmin0)
      return 0.0;
    else if (r > rcut)
      return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return -0.5 * sin((r - rmin0) * rcutfac) * rcutfac;
    }
  }
  return 0.0;
}

__device__
void compute_duarray(const int natom,
                     const int nbor,
                     const int num_atoms,
                     const int num_nbor,
                     const int twojmax,
                     const int idxdu_max,
                     const int jdimpq,
                     const int switch_flag,
                     const double x,
                     const double y,
                     const double z,
                     const double z0,
                     const double r,
                     const double dz0dr,
                     const double wj_in,
                     const double rcut,
                     const double* rootpqarray,
                     const COMPLEX* ulist,
                     COMPLEX* dulist)
{
  double r0inv;
  double a_r, a_i, b_r, b_i;
  double da_r[3], da_i[3], db_r[3], db_i[3];
  double dz0[3], dr0inv[3], dr0invdr;
  double rootpq;
  int jju, jjup, jjdu, jjdup;

  double rinv = 1.0 / r;
  double ux = x * rinv;
  double uy = y * rinv;
  double uz = z * rinv;

  r0inv = 1.0 / sqrt(r * r + z0 * z0);
  a_r = z0 * r0inv;
  a_i = -z * r0inv;
  b_r = y * r0inv;
  b_i = -x * r0inv;

  dr0invdr = -pow(r0inv, 3.0) * (r + z0 * dz0dr);

  dr0inv[0] = dr0invdr * ux;
  dr0inv[1] = dr0invdr * uy;
  dr0inv[2] = dr0invdr * uz;

  dz0[0] = dz0dr * ux;
  dz0[1] = dz0dr * uy;
  dz0[2] = dz0dr * uz;

  for (int k = 0; k < 3; k++) {
    da_r[k] = dz0[k] * r0inv + z0 * dr0inv[k];
    da_i[k] = -z * dr0inv[k];
  }

  da_i[2] += -r0inv;

  for (int k = 0; k < 3; k++) {
    db_r[k] = y * dr0inv[k];
    db_i[k] = -x * dr0inv[k];
  }

  db_i[0] += -r0inv;
  db_r[1] += r0inv;

  for (int k = 0; k < 3; ++k)
    dulist[DULIST_INDEX(natom, nbor, 0, k)] = { 0.0, 0.0 };

  jju = 1;
  jjdu = 1;
  for (int j = 1; j <= twojmax; j++) {
    int deljju = j + 1;
    for (int mb = 0; 2 * mb <= j; mb++) {

      for (int k = 0; k < 3; ++k)
        dulist[DULIST_INDEX(natom, nbor, jjdu, k)] = { 0.0, 0.0 };

      jju += deljju;
      jjdu += deljju;
    }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
  }

  jju = 1;
  jjdu = 1;
  jjup = 0;
  jjdup = 0;
  for (int j = 1; j <= twojmax; j++) {
    int deljju = j + 1;
    int deljjup = j;

    for (int mb = 0; 2 * mb < j; mb++) {

      for (int ma = 0; ma < j; ma++) {

        double up_r = ulist[ULIST_INDEX(natom, nbor, jjup)].re;
        double up_i = ulist[ULIST_INDEX(natom, nbor, jjup)].im;

        rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, j - mb)];
        for (int k = 0; k < 3; k++) {
          dulist[DULIST_INDEX(natom, nbor, jjdu, k)].re +=
            rootpq * (da_r[k] * up_r + da_i[k] * up_i +
                      a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re +
                      a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
          dulist[DULIST_INDEX(natom, nbor, jjdu, k)].im +=
            rootpq * (da_r[k] * up_i - da_i[k] * up_r +
                      a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im -
                      a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
        }

        rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, j - mb)];
        for (int k = 0; k < 3; k++) {
          dulist[DULIST_INDEX(natom, nbor, jjdu + 1, k)].re =
            -rootpq * (db_r[k] * up_r + db_i[k] * up_i +
                       b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re +
                       b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
          dulist[DULIST_INDEX(natom, nbor, jjdu + 1, k)].im =
            -rootpq * (db_r[k] * up_i - db_i[k] * up_r +
                       b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im -
                       b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
        }

        // assign middle column i.e. mb+1

        if (2 * (mb + 1) == j) {
          rootpq = rootpqarray[ROOTPQ_INDEX(j - ma, mb + 1)];
          for (int k = 0; k < 3; k++) {
            dulist[DULIST_INDEX(natom, nbor, jjdu + deljju, k)].re +=
              rootpq * (db_r[k] * up_r - db_i[k] * up_i +
                        b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re -
                        b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
            dulist[DULIST_INDEX(natom, nbor, jjdu + deljju, k)].im +=
              rootpq * (db_r[k] * up_i + db_i[k] * up_r +
                        b_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im +
                        b_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
          }

          rootpq = rootpqarray[ROOTPQ_INDEX(ma + 1, mb + 1)];
          for (int k = 0; k < 3; k++) {
            dulist[DULIST_INDEX(natom, nbor, jjdu + 1 + deljju, k)].re =
              rootpq * (da_r[k] * up_r - da_i[k] * up_i +
                        a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re -
                        a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im);
            dulist[DULIST_INDEX(natom, nbor, jjdu + 1 + deljju, k)].im =
              rootpq * (da_r[k] * up_i + da_i[k] * up_r +
                        a_r * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].im +
                        a_i * dulist[DULIST_INDEX(natom, nbor, jjdup, k)].re);
          }
        }

        jju++;
        jjup++;
        jjdu++;
        jjdup++;
      }
      jju++;
      jjdu++;
    }
    if (j % 2 == 0) {
      jju += deljju;
      jjdu += deljju;
    }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
    int ncolhalfp = deljjup / 2;
    jjup += deljjup * ncolhalfp;
  }

  double sfac = compute_sfac(r, rcut, switch_flag);
  double dsfac = compute_dsfac(r, rcut, switch_flag);

  sfac *= wj_in;
  dsfac *= wj_in;
  jju = 0;
  jjdu = 0;
  for (int j = 0; j <= twojmax; j++) {
    int deljju = j + 1;
    for (int mb = 0; 2 * mb <= j; mb++)
      for (int ma = 0; ma <= j; ma++) {
        dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].re =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].re * ux +
          sfac * dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].re;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].im =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].im * ux +
          sfac * dulist[DULIST_INDEX(natom, nbor, jjdu, 0)].im;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].re =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].re * uy +
          sfac * dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].re;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].im =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].im * uy +
          sfac * dulist[DULIST_INDEX(natom, nbor, jjdu, 1)].im;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].re =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].re * uz +
          sfac * dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].re;
        dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].im =
          dsfac * ulist[ULIST_INDEX(natom, nbor, jju)].im * uz +
          sfac * dulist[DULIST_INDEX(natom, nbor, jjdu, 2)].im;
        jju++;
        jjdu++;
      }
    int ncolhalf = deljju / 2;
    jju += deljju * ncolhalf;
  }
}

/* ----------------------------------------------------------------------
  Elapsed Time
------------------------------------------------------------------------- */
inline double elapsedTime(timeval start_time, timeval end_time)
{
  return ((end_time.tv_sec - start_time.tv_sec) +
          1e-6 * (end_time.tv_usec - start_time.tv_usec));
}

void
options(int argc, char* argv[])
{

  for (int i = 1; i < argc; i++) {

    if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
      printf("TestSNAP 1.0 (stand-alone SNAP force kernel)\n\n");
      printf("The following optional command-line switches override default "
             "values\n");
      printf("-ns, --nsteps <val>: set the number of force calls to val "
             "(default 1)\n");
      exit(0);
    } else if ((strcmp(argv[i], "-ns") == 0) ||
               (strcmp(argv[i], "--nsteps") == 0)) {
      nsteps = atoi(argv[++i]);
    } else {
      printf("ERROR: Unknown command line argument: %s\n", argv[i]);
      exit(1);
    }
  }
}



/* ----------------------------------------------------------------------
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

double factorial(int n)
{
  if (n < 0 || n > nmaxfactorial) {
    // printf("Invalid argument to factorial %d", n);
    exit(1);
  }

  return nfac_table[n];
}

