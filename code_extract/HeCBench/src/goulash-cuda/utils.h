/*
  Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
  Goulash project contributors LLNL-CODE-795383, All rights reserved.
  For details about use and distribution, please read LICENSE and NOTICE from
  the Goulash project repository: http://github.com/llnl/goulash
  SPDX-License-Identifier: BSD-3-Clause
*/


/* Get raw time in seconds as double (a large number).
 * Returns -1.0 on unexpected error.
 */
double get_raw_secs( void )
{
  struct timeval ts;
  int status;
  double raw_time;

  /* Get wall-clock time */
  /* status = getclock( CLOCK_REALTIME, &ts ); */
  status = gettimeofday( &ts, NULL );

  /* Return -1.0 on error */
  if( status != 0 ) return -1.0;

  /* Convert structure to double (in seconds ) (a large number) */
  raw_time = (double)ts.tv_sec + (double)ts.tv_usec * 1e-6;

  return (raw_time);
}

/* Returns base time.  If new_time >= 0, 
 * sets base_time to new_time before returning.
 * Using this as access method to static variable
 * in a way I can trivially emulate in fortran.
 *
 * Note: Lock shouldn't be needed, since even if multiple
 *       threads initialize this, it will be to basically
 *       the same value.
 */
double get_base_time(double new_time)
{
  static double base_time = -1.0;

  /* If passed value >= 0, use as new base_time */ 
  if (new_time >= 0.0)
    base_time = new_time;

  return(base_time);
}

/* Returns time in seconds (double) since the first call to secs_elapsed
 * (i.e., the first call returns 0.0).
 */
double secs_elapsed( void )
{
  double new_time;
  double base_time;

  /* Get current raw time (a big number) */
  new_time = get_raw_secs();

  /* Get the offset since first time called (pass -1 to query)*/
  base_time = get_base_time(-1.0);

  /* If base time not set (negative), set to current time (pass in positive secs)*/
  if (base_time < 0.0) base_time = get_base_time(new_time);

  /* Returned offset from first time called */
  return (new_time - base_time);
}

void reference(double* __restrict m_gate, const long nCells, const double* __restrict Vm) 
{
  for (long ii = 0; ii < nCells; ii++) {
    double sum1,sum2;
    const double x = Vm[ii];
    const int Mhu_l = 10;
    const int Mhu_m = 5;
    const double Mhu_a[] = { 9.9632117206253790e-01,  4.0825738726469545e-02,  6.3401613233199589e-04,  4.4158436861700431e-06,  1.1622058324043520e-08,  1.0000000000000000e+00,  4.0568375699663400e-02,  6.4216825832642788e-04,  4.2661664422410096e-06,  1.3559930396321903e-08, -1.3573468728873069e-11, -4.2594802366702580e-13,  7.6779952208246166e-15,  1.4260675804433780e-16, -2.6656212072499249e-18};

    sum1 = 0;
    for (int j = Mhu_m-1; j >= 0; j--)
      sum1 = Mhu_a[j] + x*sum1;
    sum2 = 0;
    int k = Mhu_m + Mhu_l - 1;
    for (int j = k; j >= Mhu_m; j--)
      sum2 = Mhu_a[j] + x * sum2;
    double mhu = sum1/sum2;

    const int Tau_m = 18;
    const double Tau_a[] = {1.7765862602413648e+01*0.02,  5.0010202770602419e-02*0.02, -7.8002064070783474e-04*0.02, -6.9399661775931530e-05*0.02,  1.6936588308244311e-06*0.02,  5.4629017090963798e-07*0.02, -1.3805420990037933e-08*0.02, -8.0678945216155694e-10*0.02,  1.6209833004622630e-11*0.02,  6.5130101230170358e-13*0.02, -6.9931705949674988e-15*0.02, -3.1161210504114690e-16*0.02,  5.0166191902609083e-19*0.02,  7.8608831661430381e-20*0.02,  4.3936315597226053e-22*0.02, -7.0535966258003289e-24*0.02, -9.0473475495087118e-26*0.02, -2.9878427692323621e-28*0.02,  1.0000000000000000e+00};

    sum1 = 0;
    for (int j = Tau_m-1; j >= 0; j--)
      sum1 = Tau_a[j] + x*sum1;
    double tauR = sum1;
    m_gate[ii] += (mhu - m_gate[ii])*(1-exp(-tauR));
  }
}
