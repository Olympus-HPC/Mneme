
#ifndef SNAP_H
#define SNAP_H

#include <chrono>

extern int nsteps;

using namespace std::chrono;
typedef high_resolution_clock myclock;
typedef duration<float> myduration;


// Complex type structure
typedef struct
{
  double re;
  double im;
} COMPLEX;

typedef struct 
{
  int j1; 
  int j2;
  int j;
} SNA_BINDICES;

double factorial(int);
double deltacg(int, int, int);
int compute_ncoeff(int twojmax);
void options(int argc, char* argv[]);
inline double elapsedTime(timeval start_time, timeval end_time);

__device__
double compute_sfac(double, double, int);
__device__
double compute_dsfac(double, double, int);
__device__
void compute_duarray(
    const int natom,
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
    COMPLEX* dulist);

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MY_PI  3.14159265358979323846
#define nmaxfactorial 167
#define rfac0  0.99363
#define rmin0  0.0

#define  DULIST_INDEX(a, b, c, d) \
  ((a) + num_atoms * (b) + num_atoms * num_nbor * (c) + num_atoms * num_nbor * idxdu_max * (d))

#define  ULIST_INDEX(a, b, c) \
  ((a) + num_atoms * (b) + num_atoms * num_nbor * (c))

#define ROOTPQ_INDEX(a, b) ((a) + jdimpq * (b))

#define  INDEX_2D(a, b) ((a) + num_atoms * (b))

#define  F_INDEX(a, b) ((a) + ntotal * (b))

#define  IDXZ_INDEX(a, b) ((a) + idxz_max * (b))


#endif
