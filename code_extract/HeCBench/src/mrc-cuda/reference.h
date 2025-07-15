void reference (const int N,
                const int* Y,
                const float* X1,
                const float* X2,
                const float* dOutput,
                const float margin,
                      float* dX1,
                      float* dX2)
{
  for (int i = 0; i < N; i++) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = dX2[i] = 0.f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}
