#include <cmath>
#include <memory>

void cubicSolver_cpu(int n, float *A, float *B, float *C, float *D, float *Q,
                     float *R, float *del, float *theta, float *sqrtQ,
                     float *x1, float *x2, float *x3, float *x1_img,
                     float *x2_img, float *x3_img) {
  // solver for finding roots (x1, x2, x3) for ax^3 + bx^2 + cx + d = 0

  const float TWO_PI = 2 * 3.1415927;
  const float FOUR_PI = 4 * 3.1415927;

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {

    Q[i] = ((3 * C[i]) / A[i] - ((B[i] * B[i]) / (A[i] * A[i]))) / 9;
    R[i] = (((-(2 * (B[i] * B[i] * B[i])) / (A[i] * A[i] * A[i]))) +
            ((9 * (B[i] * C[i])) / (A[i] * A[i])) - ((27 * D[i]) / A[i])) /
           54;
    del[i] = ((R[i] * R[i])) + ((Q[i] * Q[i] * Q[i]));
  }

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {

    // all 3 roots real
    if (del[i] <= 0) {

      theta[i] = acosf((float)(R[i] / sqrtf((float)-(Q[i] * Q[i] * Q[i]))));
      sqrtQ[i] = 2 * sqrtf((float)-Q[i]);

      x1[i] = (sqrtQ[i] * cosf((float)(theta[i] / 3)) - (B[i] / (A[i] * 3)));
      x2[i] = (sqrtQ[i] * cosf((float)(theta[i] + TWO_PI) / 3)) -
              (B[i] / (A[i] * 3));
      x3[i] = (sqrtQ[i] * cosf((float)(theta[i] + FOUR_PI) / 3)) -
              (B[i] / (A[i] * 3));
    }

    if (del[i] > 0) { // only 1 real root

      //            S = (R + sqrtD)^(1 / 3);
      //            T = (R - sqrtD)^(1 / 3);
      //            x = S + T - b/3a;

      // real root

      x1[i] = ((cbrtf((float)(R[i] + sqrtf((float)del[i])))) +
               cbrtf((float)(R[i] - sqrtf((float)del[i])))) -
              (B[i] / (3 * A[i]));

      x1_img[i] = 0;

      // complex conjugate roots

      x2[i] = -((cbrtf((float)(R[i] + sqrtf((float)del[i]))) +
                 cbrtf((float)(R[i] - sqrtf((float)del[i])))) /
                2) -
              (B[i] / (3 * A[i]));

      x2_img[i] = ((sqrtf((float)3) / 2) *
                   (cbrtf((float)(R[i] + sqrtf((float)del[i]))) -
                    cbrtf((float)(R[i] - sqrtf((float)del[i])))));

      x3[i] = x2[i];

      x3_img[i] = -x2_img[i];
    }

    if (Q[i] == 0 && R[i] == 0) { // all roots real and equal

      x1[i] = -(B[i] / 3);
      x2[i] = x1[i];
      x3[i] = x1[i];
    }
  }
}

void quarticSolver_cpu(int n, float *A, float *B, float *C, float *D, float *b,
                       float *c, float *d, float *Q, float *R, float *Qint,
                       float *Rint, float *del, float *theta, float *sqrtQ,
                       float *x1, float *x2, float *x3, float *temp,
                       float *min) {
  // solver for finding minimum (xmin) for f(x) = Ax^4 + Bx^3 + Cx^2 + Dx + E
  // undefined behaviour if A=0

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {

    b[i] = 0.75 * (B[i] / A[i]);
    c[i] = 0.50 * (C[i] / A[i]);
    d[i] = 0.25 * (D[i] / A[i]);

    Q[i] = (c[i] / 3) - ((b[i] * b[i]) / 9);
    R[i] = (b[i] * c[i]) / 6 - (b[i] * b[i] * b[i]) / 27 - 0.5 * d[i];

    // round Q and R to get around problems caused by floating point precision
    Q[i] = roundf(Q[i] * 1E5) / 1E5;
    R[i] = roundf(R[i] * 1E5) / 1E5;

    Qint[i] = (Q[i] * Q[i] * Q[i]);
    Rint[i] = (R[i] * R[i]);

    del[i] = Rint[i] + Qint[i];
    // del[i] = (R[i] * R[i]) + (Q[i] * Q[i] * Q[i]); // why not just Q*Q*Q +
    // R*R? Heisenbug. Heisenbug in release code
  }
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    // comparing against 1E-5 to deal with potential problems with comparing
    // floats to zero
    if (del[i] <= 1E-5) { // all 3 roots real

      theta[i] = acosf((float)(R[i] / sqrtf((float)-(Q[i] * Q[i] * Q[i]))));
      sqrtQ[i] = 2 * sqrtf((float)-Q[i]);

      x1[i] = ((sqrtQ[i] * cosf((float)(theta[i]) / 3)) - (b[i] / 3));
      x2[i] = ((sqrtQ[i] * cosf((float)(theta[i] + 2 * 3.1415927) / 3)) -
               (b[i] / 3));
      x3[i] = ((sqrtQ[i] * cosf((float)(theta[i] + 4 * 3.1415927) / 3)) -
               (b[i] / 3));

      // unrolled bubble sort  // this vs CUDA sort??
      if (x1[i] < x2[i]) {
        temp[i] = x1[i];
        x1[i] = x2[i];
        x2[i] = temp[i];
      } // { swap(x1[i], x2[i]); }//swap
      if (x2[i] < x3[i]) {
        temp[i] = x2[i];
        x2[i] = x3[i];
        x3[i] = temp[i];
      } //{ swap(x2[i], x3[i]); }//swap
      if (x1[i] < x2[i]) {
        temp[i] = x1[i];
        x1[i] = x2[i];
        x2[i] = temp[i];
      } //{ swap(x1[i], x2[i]); }//swap

      min[i] =
          A[i] *
                          ((x1[i] * x1[i] * x1[i] * x1[i]) -
                           (x3[i] * x3[i] * x3[i] * x3[i])) /
                          4 +
                      B[i] *
                          ((x1[i] * x1[i] * x1[i]) - (x3[i] * x3[i] * x3[i])) /
                          3 +
                      C[i] * ((x1[i] * x1[i]) - (x3[i] * x3[i])) / 2 +
                      D[i] * (x1[i] - x3[i]) <=
                  0
              ? x1[i]
              : x3[i];
    } else {

      x1[i] = cbrtf((float)(R[i] + sqrtf((float)del[i]))) +
              cbrtf((float)(R[i] - sqrtf((float)del[i]))) -
              (b[i] / 3); // real root

      // complex conjugate roots not relevant for minimisation

      x2[i] = 0;
      x3[i] = 0;

      min[i] = x1[i];
    }
  }
}

void QuarticMinimumCPU(int N, float *A, float *B, float *C, float *D, float *E,
                       float *min_cpu) {

  auto Q = std::make_unique<float[]>(N);
  auto R = std::make_unique<float[]>(N);
  auto Qint = std::make_unique<float[]>(N);
  auto Rint = std::make_unique<float[]>(N);
  auto del = std::make_unique<float[]>(N);

  auto bi = std::make_unique<float[]>(N);
  auto ci = std::make_unique<float[]>(N);
  auto di = std::make_unique<float[]>(N);
  auto h_temp = std::make_unique<float[]>(N);

  auto h_theta = std::make_unique<float[]>(N);
  auto h_sqrtQ = std::make_unique<float[]>(N);

  auto x1_cpu = std::make_unique<float[]>(N);
  auto x2_cpu = std::make_unique<float[]>(N);
  auto x3_cpu = std::make_unique<float[]>(N);

  quarticSolver_cpu(N, A, B, C, D, bi.get(), ci.get(), di.get(), Q.get(),
                    R.get(), Qint.get(), Rint.get(), del.get(), h_theta.get(),
                    h_sqrtQ.get(), x1_cpu.get(), x2_cpu.get(), x3_cpu.get(),
                    h_temp.get(), min_cpu);
}
