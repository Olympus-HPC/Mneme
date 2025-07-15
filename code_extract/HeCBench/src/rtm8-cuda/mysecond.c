/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.

   You will need to compile with "-DUNDERSCORE"
   to get this to link with FORTRAN on many systems.
*/

#include <sys/time.h>
/* int gettimeofday(struct timeval *tp, struct timezone *tzp); */

#ifdef UNDERSCORE
double mysecond_()
#else
double mysecond()
#endif
{
  struct timeval tp;
  struct timezone tzp;

  gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
