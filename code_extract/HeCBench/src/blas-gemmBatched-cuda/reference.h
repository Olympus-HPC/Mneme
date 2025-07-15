// B and C are vectors
template <typename T>
void gemmBatched_ref (int batchCount,
                      const int M_upper, const int K_upper, const int N_upper,
                      const int M, const int K, const int N,
                      const T alpha, const T beta,
                      const T* A, int lda, T* B, int ldb, T* C, int ldc)
{
  for (int p = 0; p < batchCount; ++p) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        T c_mnp = 0;
        for (int k = 0; k < K; ++k)
          c_mnp += A[p * M_upper * K_upper + m + k*lda] * B[p * K_upper * N_upper + k + n*ldb];
        C[p * M_upper * N_upper + m + n*ldc] = alpha*c_mnp + beta*C[p * M_upper * N_upper + m + n*ldc];
      }
    }
  }
}

void performance (int m, int n, int k, double avg_time, bool is_integer = false) {
  double total_ops = double(m) * double(n) * double(k) * 2;
  double perf = total_ops / avg_time;
  auto scale_string = "G";
  auto unit_string = is_integer ? "OP/s" : "FLOP/s";

  if (perf >= 1000) {
    perf /= 1000;
    scale_string = "T";
  }

  std::cout << perf << " " << scale_string << unit_string << std::endl;
}

