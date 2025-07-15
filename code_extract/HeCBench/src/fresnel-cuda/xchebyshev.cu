////////////////////////////////////////////////////////////////////////////////
// File: xchebyshev_Tn_series.c                                               //
// Routine(s):                                                                //
//    xChebyshev_Tn_Series                                                    //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// double xChebyshev_Tn_Series(double x, double a[],int degree)//
//                                                                            //
//  Description:                                                              //
//     This routine uses Clenshaw's recursion algorithm to evaluate a given   //
//     polynomial p(x) expressed as a linear combination of Chebyshev         //
//     polynomials of the first kind, Tn, at a point x,                       //
//       p(x) = a[0] + a[1]*T[1](x) + a[2]*T[2](x) + ... + a[deg]*T[deg](x).  //
//                                                                            //
//     Clenshaw's recursion formula applied to Chebyshev polynomials of the   //
//     first kind is:                                                         //
//     Set y[degree + 2] = 0, y[degree + 1] = 0, then for k = degree, ..., 1  //
//     set y[k] = 2 * x * y[k+1] - y[k+2] + a[k].  Finally                    //
//     set y[0] = x * y[1] - y[2] + a[0].  Then p(x) = y[0].                  //
//                                                                            //
//  Arguments:                                                                //
//     double x                                                          //
//        The point at which to evaluate the polynomial.                      //
//     double a[]                                                        //
//        The coefficients of the expansion in terms of Chebyshev polynomials,//
//        i.e. a[k] is the coefficient of T[k](x).  Note that in the calling  //
//        routine a must be defined double a[N] where N >= degree + 1.        //
//     int    degree                                                          //
//        The degree of the polynomial p(x).                                  //
//                                                                            //
//  Return Value:                                                             //
//     The value of the polynomial at x.                                      //
//     If degree is negative, then 0.0 is returned.                           //
//                                                                            //
//  Example:                                                                  //
//     double x, a[N], p;                                                //
//     int    deg = N - 1;                                                    //
//                                                                            //
//     ( code to initialize x, and a[i] i = 0, ... , a[deg] )                 //
//                                                                            //
//     p = xChebyshev_Tn_Series(x, a, deg);                                   //
////////////////////////////////////////////////////////////////////////////////

extern "C" __host__ __device__
double xChebyshev_Tn_Series(double x, double a[], int degree)
{
  double yp2 = 0.0;
  double yp1 = 0.0;
  double y = 0.0;
  double two_x = x + x;
  int k;

  // Check that degree >= 0.  If not, then return 0. //

  if ( degree < 0 ) return 0.0;

  // Apply Clenshaw's recursion save the last iteration. //

  for (k = degree; k >= 1; k--, yp2 = yp1, yp1 = y) 
    y = two_x * yp1 - yp2 + a[k];

  // Now apply the last iteration and return the result. //

  return x * yp1 - yp2 + a[0];
}
