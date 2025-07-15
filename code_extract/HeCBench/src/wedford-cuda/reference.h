template <typename scalar_t, typename accscalar_t, typename outscalar_t>
void welford_reference (
      const scalar_t* __restrict__ input,
      outscalar_t* __restrict__ out_mean,
      outscalar_t* __restrict__ out_var_biased,
      const int bs, // batch size
      const int fs, // feature size
      const int ss) // spatial size
{
  #pragma omp parallel for
  for (int fid = 0; fid < fs; fid++) {
    accscalar_t naive_sum = 0, naive_sum_square = 0;
    #pragma omp parallel for collapse(2) reduction(+:naive_sum,naive_sum_square)
    for (int bid = 0; bid < bs; bid++) {
      for (int s = 0; s < ss; s++) {
        auto x_n = static_cast<accscalar_t>(input[bid * ss * fs + fid * ss + s]);
        naive_sum += x_n;
        naive_sum_square += x_n * x_n;
      }
    }
    accscalar_t naive_mean = naive_sum / (bs * ss);
    out_mean[fid] = static_cast<outscalar_t>(naive_mean);
    out_var_biased[fid] = static_cast<outscalar_t>(naive_sum_square / (bs * ss) - naive_mean * naive_mean);
  }
}
