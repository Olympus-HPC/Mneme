#include "kernel.h"

__global__ void kernel0(struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y, float alpha_im, float alpha_re, float beta_im, float beta_re)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    float private_var5_Re;
    float private_var5_Im;
    float private_var2_Re;
    float private_var3_Im;
    float private_var2_Im;
    float private_var4_Im;
    float private_var4_Re;
    float private_var3_Re;
    float private_var99_Re;
    float private_var98_Im;
    float private_var97_Im;
    float private_var99_Im;
    float private_var97_Re;
    float private_var98_Re;

    for (int c1 = 0; c1 <= ppcg_min(368, 32 * b0 + 30); c1 += 32) {
      if (32 * b0 + t0 <= 369 && c1 == 0) {
        private_var5_Re = ((Y[32 * b0 + t0].Re * beta_re) - (Y[32 * b0 + t0].Im * beta_im));
        private_var5_Im = ((Y[32 * b0 + t0].Im * beta_re) + (Y[32 * b0 + t0].Re * beta_im));
        Y[32 * b0 + t0].Re = private_var5_Re;
        Y[32 * b0 + t0].Im = private_var5_Im;
        private_var2_Re = (alpha_re * AT[11872 * b0 + 371 * t0].Re);
        private_var2_Im = (alpha_im * AT[11872 * b0 + 371 * t0].Re);
        private_var3_Re = ((private_var2_Re * X[32 * b0 + t0].Re) - (private_var2_Im * X[32 * b0 + t0].Im));
        private_var3_Im = ((private_var2_Im * X[32 * b0 + t0].Re) + (private_var2_Re * X[32 * b0 + t0].Im));
        private_var4_Re = (Y[32 * b0 + t0].Re + private_var3_Re);
        private_var4_Im = (Y[32 * b0 + t0].Im + private_var3_Im);
        Y[32 * b0 + t0].Re = private_var4_Re;
        Y[32 * b0 + t0].Im = private_var4_Im;
      }
      if (32 * b0 + t0 <= 369)
        for (int c3 = 0; c3 <= ppcg_min(31, 32 * b0 + t0 - c1 - 1); c3 += 1) {
          private_var97_Re = ((alpha_re * AT[32 * b0 + t0 + 370 * c1 + 370 * c3].Re) - (alpha_im * AT[32 * b0 + t0 + 370 * c1 + 370 * c3].Im));
          private_var97_Im = ((alpha_im * AT[32 * b0 + t0 + 370 * c1 + 370 * c3].Re) + (alpha_re * AT[32 * b0 + t0 + 370 * c1 + 370 * c3].Im));
          private_var98_Re = ((private_var97_Re * X[c1 + c3].Re) - (private_var97_Im * X[c1 + c3].Im));
          private_var98_Im = ((private_var97_Im * X[c1 + c3].Re) + (private_var97_Re * X[c1 + c3].Im));
          private_var99_Re = (Y[32 * b0 + t0].Re + private_var98_Re);
          private_var99_Im = (Y[32 * b0 + t0].Im + private_var98_Im);
          Y[32 * b0 + t0].Re = private_var99_Re;
          Y[32 * b0 + t0].Im = private_var99_Im;
        }
      __syncthreads();
    }
}
__global__ void kernel1(struct ComplexFloat *AT, struct ComplexFloat *X, struct ComplexFloat *Y, float alpha_im, float alpha_re)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    float private_var96_Re;
    float private_var96_Im;
    float private_var94_Im;
    float private_var95_Im;
    float private_var94_Re;
    float private_var95_Re;

    for (int c1 = 5888 * b0; c1 <= ppcg_min(67712, 5856 * b0 + 6016); c1 += 32) {
      for (int c3 = ppcg_max(0, 5888 * b0 + 184 * t0 - c1); c3 <= ppcg_min(31, 5856 * b0 + 183 * t0 - c1 + 368); c3 += 1) {
        private_var94_Re = ((alpha_re * AT[5984 * b0 + 187 * t0 + c1 + c3 + 1].Re) - (alpha_im * (-AT[5984 * b0 + 187 * t0 + c1 + c3 + 1].Im)));
        private_var94_Im = ((alpha_im * AT[5984 * b0 + 187 * t0 + c1 + c3 + 1].Re) + (alpha_re * (-AT[5984 * b0 + 187 * t0 + c1 + c3 + 1].Im)));
        private_var95_Re = ((private_var94_Re * X[-5856 * b0 - 183 * t0 + c1 + c3 + 1].Re) - (private_var94_Im * X[-5856 * b0 - 183 * t0 + c1 + c3 + 1].Im));
        private_var95_Im = ((private_var94_Im * X[-5856 * b0 - 183 * t0 + c1 + c3 + 1].Re) + (private_var94_Re * X[-5856 * b0 - 183 * t0 + c1 + c3 + 1].Im));
        private_var96_Re = (Y[32 * b0 + t0].Re + private_var95_Re);
        private_var96_Im = (Y[32 * b0 + t0].Im + private_var95_Im);
        Y[32 * b0 + t0].Re = private_var96_Re;
        Y[32 * b0 + t0].Im = private_var96_Im;
      }
      __syncthreads();
    }
}
