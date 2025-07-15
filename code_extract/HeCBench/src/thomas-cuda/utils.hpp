/**
 *
 *   @file utils.hpp
 *
 *   @brief utils funtions
 *
 *   cuThomasBatch is a software package provided by
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
 *
 *   @author Ivan Martinez-Perez ivan.martinez@bsc.es
 *   @author Pedro Valero-Lara   pedro.valero@bsc.es
 *
 **/

#if defined( _WIN32 ) || defined( _WIN64 )
#  include <time.h>
#  include <sys/timeb.h>
#  if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#  else
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#  endif
#else
#  include <sys/time.h>
#  include <time.h>
#endif
#include <cmath>

/* ////////////////////////////////////////////////////////////////////////////
   -- Emulate gettimeofday on Windows.
 */
#if defined( _WIN32 ) || defined( _WIN64 )
#ifndef _TIMEZONE_DEFINED
#define _TIMEZONE_DEFINED

struct timezone
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
#endif

int gettimeofday(struct timeval* tv, struct timezone* tz)
{
  FILETIME         ft;
  unsigned __int64 tmpres = 0;
  static int       tzflag = 0;

  if (NULL != tv) {
    GetSystemTimeAsFileTime(&ft);
    tmpres |=  ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |=  ft.dwLowDateTime;

    /*converting file time to unix epoch*/
    tmpres /= 10;  /*convert into microseconds*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS;

    tv->tv_sec  = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }
  if (NULL != tz) {
    if (!tzflag) {
      _tzset();
      tzflag = 1;
    }
    tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime     = _daylight;
  }
  return 0;
}
#endif


/**
 *  Return time in seconds since arbitrary time in the past.
 *  Used for elapsed wall clock time computation.
 **/
double time_wtime( void )
{
  struct timeval t;
  gettimeofday( &t, NULL );
  return t.tv_sec + t.tv_usec*1e-6;
}

/**
 * Version callable from Fortran stores seconds in the parameter time.
 **/
void timef_wtime(double *time)
{
  *time = time_wtime();
}

template <typename T>
void calcError(T* src,T* dst, int size){
  double error=0;
  for (int i = 0; i < size; ++i)
  {
    if (error < std::fabs(std::fabs(src[i]) - std::fabs(dst[i])))
    {
      error = std::fabs(std::fabs(src[i]) - std::fabs(dst[i]));
    }
  }
  printf("Maximum error: %e\n",error);
}

template <typename T>
void calcError_print(T* src,T* dst, int size){

  double error=0;

  for (int i = 0; i < size; ++i)
  {
    printf("src: %e - dst: %e\n",src[i],dst[i]);

    if (error < std::fabs(std::fabs(src[i]) - std::fabs(dst[i])))
    {
      error = std::fabs(std::fabs(src[i]) - std::fabs(dst[i]));
    }
  }
  printf("Maximum error: %e\n",error);
}
