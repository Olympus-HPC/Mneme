#define LOOP(index, range) \
   for (int index = 0; index < range; index++) 

template <typename T>
void sequenceMaskKernel_cpu(
    int N,
    int M,
    int B,
    const T* in,
    const int* seq_lengths,
    T fill_val,
    T* out)
{
  if (B >= 0) {
    #pragma omp parallel for
    LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);
      int ind = N * M * i + M * j + k;
      out[ind] = (k >= seq_lengths[j] ? fill_val : in[ind]);
    }
  } else {
    #pragma omp parallel for
    LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;
      out[index] = (j >= seq_lengths[i] ? fill_val : in[index]);
    }
  }
}

template <typename T>
void windowMaskKernel_cpu(
    int N,
    int M,
    int B,
    const T* in,
    const int* window_centers,
    const int radius,
    T fill_val,
    T* out) {
  if (B >= 0) {
    #pragma omp parallel for
    LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] =
          (k < window_centers[j] - radius || k > window_centers[j] + radius
               ? fill_val
               : in[ind]);
    }
  } else {
    #pragma omp parallel for
    LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] =
          (j < window_centers[i] - radius || j > window_centers[i] + radius
               ? fill_val
               : in[index]);
    }
  }
}

template <typename T>
void
upperMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp parallel for
    LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k > j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp parallel for
    LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j > i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void
lowerMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp parallel for
    LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k < j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp parallel for
    LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j < i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void
upperDiagMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp parallel for
    LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k >= j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp parallel for
    LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j >= i ? fill_val : in[index]);
    }
  }
}

template <typename T>
void
lowerDiagMaskKernel_cpu(int N, int M, int B, const T* in, T fill_val, T* out) {
  if (B >= 0) {
    #pragma omp parallel for
    LOOP(index, B * N * M) {
      int k = index % M;
      int j = (index - k) / M % N;
      int i = (index - M * j - k) / (N * M);

      int ind = N * M * i + M * j + k;
      out[ind] = (k <= j ? fill_val : in[ind]);
    }
  } else {
    #pragma omp parallel for
    LOOP(index, N * M) {
      int i = index / M;
      int j = index % M;

      out[index] = (j <= i ? fill_val : in[index]);
    }
  }
}


