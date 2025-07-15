////////////////////////////////////////////////////////////////////////////////
// File: fresnel_auxiliary_sine_integral.c                                    //
// Routine(s):                                                                //
//    Fresnel_Auxiliary_Sine_Integral                                         //
//    xFresnel_Auxiliary_Sine_Integral                                        //
////////////////////////////////////////////////////////////////////////////////

#include <math.h>           // required for fabs()                      
#include <float.h>          // required for DBL_EPSILON

//                         Externally Defined Routines                        //
extern "C" __host__ __device__
double xChebyshev_Tn_Series(double x, const double a[], int degree);

//                         Internally Defined Routines                        //
__host__ __device__
double      Fresnel_Auxiliary_Sine_Integral( double x );
__host__ __device__
double xFresnel_Auxiliary_Sine_Integral( double x );

__host__ __device__
static double Chebyshev_Expansion_0_1(double x);
__host__ __device__
static double Chebyshev_Expansion_1_3(double x);
__host__ __device__
static double Chebyshev_Expansion_3_5(double x);
__host__ __device__
static double Chebyshev_Expansion_5_7(double x);
__host__ __device__
static double Asymptotic_Series( double x );

//                         Internally Defined Constants                       //
static double const sqrt_2pi = 2.506628274631000502415765284811045253006;

////////////////////////////////////////////////////////////////////////////////
// double xFresnel_Auxiliary_Sine_Integral( double x )                   //
//                                                                            //
//  Description:                                                              //
//     The Fresnel auxiliary sine integral, g(x), is the integral from 0 to   //
//     infinity of the integrand                                              //
//                     sqrt(2/pi) exp(-2xt) sin(t^2) dt                       //
//     where x >= 0.                                                          //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary sine integral    //
//                     g() where x >= 0.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary sine integral g evaluated at        //
//     x >= 0.                                                                //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                           //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = xFresnel_Auxiliary_Sine_Integral( x );                             //
////////////////////////////////////////////////////////////////////////////////
__host__ __device__
double xFresnel_Auxiliary_Sine_Integral( double x )
{
   if (x == 0.0) return 0.5;
   if (x <= 1.0) return Chebyshev_Expansion_0_1(x);
   if (x <= 3.0) return Chebyshev_Expansion_1_3(x);
   if (x <= 5.0) return Chebyshev_Expansion_3_5(x);
   if (x <= 7.0) return Chebyshev_Expansion_5_7(x);
   return Asymptotic_Series( x );
}


////////////////////////////////////////////////////////////////////////////////
// static double Chebyshev_Expansion_0_1( double x )                //
//                                                                            //
//  Description:                                                              //
//     Evaluate the Fresnel auxiliary sine integral, g(x), on the interval    //
//     0 < x <= 1 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary sine integral    //
//                     where 0 < x <= 1.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary sine integral g evaluated at        //
//     x where 0 < x <= 1.                                                    //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                      //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = Chebyshev_Expansion_0_1(x);                                        //
////////////////////////////////////////////////////////////////////////////////

__host__ __device__
static double Chebyshev_Expansion_0_1( double x )
{ 
   static double const c[] = {
      +2.560134650043040830997e-1,  -1.993005146464943284549e-1,
      +4.025503636721387266117e-2,  -4.459600454502960250729e-3,
      +6.447097305145147224459e-5,  +7.544218493763717599380e-5,
      -1.580422720690700333493e-5,  +1.755845848573471891519e-6,
      -9.289769688468301734718e-8,  -5.624033192624251079833e-9,
      +1.854740406702369495830e-9,  -2.174644768724492443378e-10,
      +1.392899828133395918767e-11, -6.989216003725983789869e-14,
      -9.959396121060010838331e-14, +1.312085140393647257714e-14,
      -9.240470383522792593305e-16, +2.472168944148817385152e-17,
      +2.834615576069400293894e-18, -4.650983461314449088349e-19,
      +3.544083040732391556797e-20
   };

   static const int degree = sizeof(c) / sizeof(double) - 1;
   static const double midpoint = 0.5;
   static const double scale = 0.5;
   
   return xChebyshev_Tn_Series( (x - midpoint) / scale, c, degree );
}


////////////////////////////////////////////////////////////////////////////////
// static double Chebyshev_Expansion_1_3( double x )                //
//                                                                            //
//  Description:                                                              //
//     Evaluate the Fresnel auxiliary sine integral, g(x), on the interval    //
//     1 < x <= 3 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary sine integral    //
//                     where 1 < x <= 3.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary sine integral g evaluated at        //
//     x where 1 < x <= 3.                                                    //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                      //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = Chebyshev_Expansion_1_3(x);                                        //
////////////////////////////////////////////////////////////////////////////////

__host__ __device__
static double Chebyshev_Expansion_1_3( double x )
{ 
   static double const c[] = {
      +3.470341566046115476477e-2,  -3.855580521778624043304e-2,
      +1.420604309383996764083e-2,  -4.037349972538938202143e-3,
      +9.292478174580997778194e-4,  -1.742730601244797978044e-4,
      +2.563352976720387343201e-5,  -2.498437524746606551732e-6,
      -1.334367201897140224779e-8,  +7.436854728157752667212e-8,
      -2.059620371321272169176e-8,  +3.753674773239250330547e-9,
      -5.052913010605479996432e-10, +4.580877371233042345794e-11,
      -7.664740716178066564952e-13, -7.200170736686941995387e-13,
      +1.812701686438975518372e-13, -2.799876487275995466163e-14,
      +3.048940815174731772007e-15, -1.936754063718089166725e-16,
      -7.653673328908379651914e-18, +4.534308864750374603371e-18,
      -8.011054486030591219007e-19, +9.374587915222218230337e-20,
      -7.144943099280650363024e-21, +1.105276695821552769144e-22,
      +6.989334213887669628647e-23 
   };

   static const int degree = sizeof(c) / sizeof(double) - 1;
   static const double midpoint = 2.0;
   
   return xChebyshev_Tn_Series( (x - midpoint), c, degree );
}


////////////////////////////////////////////////////////////////////////////////
// static double Chebyshev_Expansion_3_5( double x )                //
//                                                                            //
//  Description:                                                              //
//     Evaluate the Fresnel auxiliary sine integral, g(x), on the interval    //
//     3 < x <= 5 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary sine integral    //
//                     where 3 < x <= 5.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary sine integral g evaluated at        //
//     x where 3 < x <= 5.                                                    //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                      //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = Chebyshev_Expansion_3_5(x);                                        //
////////////////////////////////////////////////////////////////////////////////

__host__ __device__
static double Chebyshev_Expansion_3_5( double x )
{ 
   static double const c[] = {
      +3.684922395955255848372e-3,  -2.624595437764014386717e-3,
      +6.329162500611499391493e-4,  -1.258275676151483358569e-4,
      +2.207375763252044217165e-5,  -3.521929664607266176132e-6,
      +5.186211398012883705616e-7,  -7.095056569102400546407e-8,
      +9.030550018646936241849e-9,  -1.066057806832232908641e-9,
      +1.157128073917012957550e-10, -1.133877461819345992066e-11,
      +9.633572308791154852278e-13, -6.336675771012312827721e-14,
      +1.634407356931822107368e-15, +3.944542177576016972249e-16,
      -9.577486627424256130607e-17, +1.428772744117447206807e-17,
      -1.715342656474756703926e-18, +1.753564314320837957805e-19,
      -1.526125102356904908532e-20, +1.070275366865736879194e-21,
      -4.783978662888842165071e-23
   };

   static const int degree = sizeof(c) / sizeof(double) - 1;
   static const double midpoint = 4.0;
   
   return xChebyshev_Tn_Series( (x - midpoint), c, degree );
}


////////////////////////////////////////////////////////////////////////////////
// static double Chebyshev_Expansion_5_7( double x )                //
//                                                                            //
//  Description:                                                              //
//     Evaluate the Fresnel auxiliary sine integral, g(x), on the interval    //
//     5 < x <= 7 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary sine integral    //
//                     where 5 < x <= 7.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary sine integral g evaluated at        //
//     x where 5 < x <= 7.                                                    //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                      //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = Chebyshev_Expansion_5_7(x);                                        //
////////////////////////////////////////////////////////////////////////////////

__host__ __device__
static double Chebyshev_Expansion_5_7( double x )
{ 
   static double const c[] = {
      +1.000801217561417083840e-3,  -4.915205279689293180607e-4,
      +8.133163567827942356534e-5,  -1.120758739236976144656e-5,
      +1.384441872281356422699e-6,  -1.586485067224130537823e-7,
      +1.717840749804993618997e-8,  -1.776373217323590289701e-9,
      +1.765399783094380160549e-10, -1.692470022450343343158e-11,
      +1.568238301528778401489e-12, -1.405356860742769958771e-13,
      +1.217377701691787512346e-14, -1.017697418261094517680e-15,
      +8.186068056719295045596e-17, -6.305153620995673221364e-18,
      +4.614110100197028845266e-19, -3.165914620159266813849e-20,
      +1.986716456911232767045e-21, -1.078418278174434671506e-22,
      +4.255983404468350776788e-24
   };

   static const int degree = sizeof(c) / sizeof(double) - 1;
   static const double midpoint = 6.0;
   
   return xChebyshev_Tn_Series( (x - midpoint), c, degree );

}


////////////////////////////////////////////////////////////////////////////////
// static double Asymptotic_Series( double x )                      //
//                                                                            //
//  Description:                                                              //
//     For a large argument x, the auxiliary Fresnel sine integral, g(x),     //
//     can be expressed as the asymptotic series                              //
//      g(x) ~ 1/(x^3 * sqrt(8pi))[1 - 15/4x^4 + 945/16x^8 + ... +            //
//                                                (4j+1)!!/(-4x^4)^j + ... ]  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary sine integral    //
//                     where x > 7.                                           //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary sine integral g evaluated at        //
//     x where x > 7.                                                         //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                      //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = Asymptotic_Series( x );                                            //
////////////////////////////////////////////////////////////////////////////////
#define NUM_ASYMPTOTIC_TERMS 35
__host__ __device__
static double Asymptotic_Series( double x )
{
   double x2 = x * x;
   double x4 = -4.0 * x2 * x2;
   double xn = 1.0;
   double factorial = 1.0;
   double g = 0.0;
   double term[NUM_ASYMPTOTIC_TERMS + 1];
   double epsilon = DBL_EPSILON / 4.0;
   int j = 5;
   int i = 0;

   term[0] = 1.0;
   term[NUM_ASYMPTOTIC_TERMS] = 0.0;
   for (i = 1; i < NUM_ASYMPTOTIC_TERMS; i++) {
      factorial *= ( (double)j * (double)(j - 2));
      xn *= x4;
      term[i] = factorial / xn;
      j += 4;
      if (fabs(term[i]) >= fabs(term[i-1])) {
         i--;
         break;
      }
      if (fabs(term[i]) <= epsilon) break;
   }
   for (; i >= 0; i--) g += term[i];

   g /= ( x * sqrt_2pi);
   return g / (x2 + x2);
}
