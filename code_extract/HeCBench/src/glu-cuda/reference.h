// Return product of all dimensions starting from k
inline int64_t size_from_dim(int k, std::vector<int> &dims) {
  int64_t r = 1;
  for (uint64_t i = k; i < dims.size(); i++) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline int64_t size_to_dim(int k, std::vector<int> &dims) {
  int64_t r = 1;
  for (int i = 0; i < k; i++) {
    r *= dims[i];
  }
  return r;
}

float sigmoid(const float x) {
  if (x >= 0) {
    return 1.f / (1.f + expf(-x));
  } else {
    const float exp_x = expf(x);
    return exp_x / (1 + exp_x);
  }
}

void ComputeGlu(
    const int M,
    const int split_dim,
    const int N,
    const float* Xdata,
    float* Ydata)
{
  const int yStride = split_dim * N;
  const int xStride = 2 * yStride;
  for (int i = 0; i < M; ++i) {
    const int idx = i * xStride;
    const int idy = i * yStride;
    for (int j = 0; j < split_dim; ++j) {
      const int jN = j * N;
      const int jdx1 = idx + jN;
      const int jdx2 = idx + (j + split_dim) * N;
      const int jdy = idy + jN;
      for (int k = 0; k < N; ++k) {
        const float x1 = Xdata[jdx1 + k];
        const float x2 = Xdata[jdx2 + k];
        Ydata[jdy + k] = x1 * sigmoid(x2);
      }
    }
  }
}

