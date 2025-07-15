////////////////////////////////////////////////////////////////////////////////
// File: fresnel_auxiliary_cosine_integral.c                                  //
// Routine(s):                                                                //
//    Fresnel_Auxiliary_Cosine_Integral                                       //
//    xFresnel_Auxiliary_Cosine_Integral                                      //
////////////////////////////////////////////////////////////////////////////////

#include <math.h>           // required for fabs()
#include <float.h>          // required for DBL_EPSILON

//                         Externally Defined Routines                        //
extern "C" __host__ __device__
double xChebyshev_Tn_Series(double x, const double a[], int degree);

//                         Internally Defined Routines                        //
__host__ __device__
double      Fresnel_Auxiliary_Cosine_Integral( double x );
__host__ __device__
double xFresnel_Auxiliary_Cosine_Integral( double x );

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
// double xFresnel_Auxiliary_Cosine_Integral( double x )                 //
//                                                                            //
//  Description:                                                              //
//     The Fresnel auxiliary cosine integral, f(x), is the integral from 0 to //
//     infinity of the integrand                                              //
//                     sqrt(2/pi) exp(-2xt) cos(t^2) dt                       //
//     where x >= 0.                                                          //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary cosine integral  //
//                     f() where x >= 0.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary cosine integral f evaluated at      //
//     x >= 0.                                                                //
//                                                                            //
//  Example:                                                                  //
//     double y, x;                                                           //
//                                                                            //
//     ( code to initialize x )                                               //
//                                                                            //
//     y = xFresnel_Auxiliary_Cosine_Integral( x );                           //
////////////////////////////////////////////////////////////////////////////////
__host__ __device__
double xFresnel_Auxiliary_Cosine_Integral( double x )
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
//     Evaluate the Fresnel auxiliary cosine integral, f(x), on the interval  //
//     0 < x <= 1 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary cosine integral  //
//                     where 0 < x <= 1.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary cosine integral f evaluated at      //
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
      +4.200987560240514577713e-1,  -9.358785913634965235904e-2,
      -7.642539415723373644927e-3,  +4.958117751796130135544e-3,
      -9.750236036106120253456e-4,  +1.075201474958704192865e-4,
      -4.415344769301324238886e-6,  -7.861633919783064216022e-7,
      +1.919240966215861471754e-7,  -2.175775608982741065385e-8,
      +1.296559541430849437217e-9,  +2.207205095025162212169e-11,
      -1.479219615873704298874e-11, +1.821350127295808288614e-12,
      -1.228919312990171362342e-13, +2.227139250593818235212e-15,
      +5.734729405928016301596e-16, -8.284965573075354177016e-17,
      +6.067422701530157308321e-18, -1.994908519477689596319e-19,
      -1.173365630675305693390e-20
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
//     Evaluate the Fresnel auxiliary cosine integral, f(x), on the interval  //
//     1 < x <= 3 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary cosine integral  //
//                     where 1 < x <= 3.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary cosine integral f evaluated at      //
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
      +2.098677278318224971989e-1,  -9.314234883154103266195e-2,
      +1.739905936938124979297e-2,  -2.454274824644285136137e-3,
      +1.589872606981337312438e-4,  +4.203943842506079780413e-5,
      -2.018022256093216535093e-5,  +5.125709636776428285284e-6,
      -9.601813551752718650057e-7,  +1.373989484857155846826e-7,
      -1.348105546577211255591e-8,  +2.745868700337953872632e-10,
      +2.401655517097260106976e-10, -6.678059547527685587692e-11,
      +1.140562171732840809159e-11, -1.401526517205212219089e-12,
      +1.105498827380224475667e-13, +2.040731455126809208066e-16,
      -1.946040679213045143184e-15, +4.151821375667161733612e-16,
      -5.642257647205149369594e-17, +5.266176626521504829010e-18,
      -2.299025577897146333791e-19, -2.952226367506641078731e-20,
      +8.760405943193778149078e-21
   };

   static const int degree = sizeof(c) / sizeof(double) - 1;
   static const double midpoint = 2.0;
   
   return xChebyshev_Tn_Series( (x - midpoint), c, degree );

}


////////////////////////////////////////////////////////////////////////////////
// static double Chebyshev_Expansion_3_5( double x )                //
//                                                                            //
//  Description:                                                              //
//     Evaluate the Fresnel auxiliary cosine integral, g(x), on the interval  //
//     3 < x <= 5 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary cosine integral  //
//                     where 3 < x <= 5.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary cosine integral f evaluated at      //
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
      +1.025703371090289562388e-1,  -2.569833023232301400495e-2,
      +3.160592981728234288078e-3,  -3.776110718882714758799e-4,
      +4.325593433537248833341e-5,  -4.668447489229591855730e-6,
      +4.619254757356785108280e-7,  -3.970436510433553795244e-8,
      +2.535664754977344448598e-9,  -2.108170964644819803367e-11,
      -2.959172018518707683013e-11, +6.727219944906606516055e-12,
      -1.062829587519902899001e-12, +1.402071724705287701110e-13,
      -1.619154679722651005075e-14, +1.651319588396970446858e-15,
      -1.461704569438083772889e-16, +1.053521559559583268504e-17,
      -4.760946403462515858756e-19, -1.803784084922403924313e-20,
      +7.873130866418738207547e-21
   };

   static const int degree = sizeof(c) / sizeof(double) - 1;
   static const double midpoint = 4.0;
   
   return xChebyshev_Tn_Series( (x - midpoint), c, degree );
}


////////////////////////////////////////////////////////////////////////////////
// static double Chebyshev_Expansion_5_7( double x )                //
//                                                                            //
//  Description:                                                              //
//     Evaluate the Fresnel auxiliary cosine integral, g(x), on the interval  //
//     5 < x <= 7 using the Chebyshev interpolation formula.                  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary cosine integral  //
//                     where 5 < x <= 7.                                      //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary cosine integral f evaluated at      //
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
      +6.738667333400589274018e-2,  -1.128146832637904868638e-2,
      +9.408843234170404670278e-4,  -7.800074103496165011747e-5,
      +6.409101169623350885527e-6,  -5.201350558247239981834e-7,
      +4.151668914650221476906e-8,  -3.242202015335530552721e-9,
      +2.460339340900396789789e-10, -1.796823324763304661865e-11,
      +1.244108496436438952425e-12, -7.950417122987063540635e-14,
      +4.419142625999150971878e-15, -1.759082736751040110146e-16,
      -1.307443936270786700760e-18, +1.362484141039320395814e-18,
      -2.055236564763877250559e-19, +2.329142055084791308691e-20,
      -2.282438671525884861970e-21
   };

   static const int degree = sizeof(c) / sizeof(double) - 1;
   static const double midpoint = 6.0;
   
   return xChebyshev_Tn_Series( (x - midpoint), c, degree );

}


////////////////////////////////////////////////////////////////////////////////
// static double Asymptotic_Series( double x )                      //
//                                                                            //
//  Description:                                                              //
//     For a large argument x, the auxiliary Fresnel cosine integral, f(x),   //
//     can be expressed as the asymptotic series                              //
//      f(x) ~ 1/(x*sqrt(2pi))[1 - 3/4x^4 + 105/16x^8 + ... +                 //
//                                                (4j-1)!!/(-4x^4)^j + ... ]  //
//                                                                            //
//  Arguments:                                                                //
//     double  x  The argument of the Fresnel auxiliary cosine integral  //
//                     where x > 7.                                           //
//                                                                            //
//  Return Value:                                                             //
//     The value of the Fresnel auxiliary cosine integral f evaluated at      //
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
   double f = 0.0;
   double term[NUM_ASYMPTOTIC_TERMS + 1];
   double epsilon = DBL_EPSILON / 4.0;
   int j = 3;
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
   
   for (; i >= 0; i--) f += term[i];

   return f / (x * sqrt_2pi);   
}
