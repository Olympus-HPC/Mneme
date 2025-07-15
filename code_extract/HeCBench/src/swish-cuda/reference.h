template <typename T>
void reference (const int N, T *X, T *Y, T *dX, T *dY) {
  for(int i = 0; i < N; i++) {
    Y[i] = *(X + i) / (T(1) + exp(-*(X + i)));
    dX[i] = *(dY + i) *
            (*(Y + i) + (T(1) - *(Y + i)) / (T(1) + exp(-*(X + i))));
  }
}
