template <typename value_idx, typename value_t>
void reference(const value_t* __restrict distances,
    value_t* __restrict P,
    const float perplexity,
    const int epochs,
    const float tol,
    const value_idx n,
    const int k)
{
  const float desired_entropy = logf(perplexity);
  for (int i = 0; i < n; i++) {

    value_t beta_min = -INFINITY, beta_max = INFINITY;
    value_t beta = 1;
    const int ik = i * k;
    int step;

    for (step = 0; step < epochs; step++) {
      value_t sum_Pi = FLT_EPSILON;

      // Exponentiate to get Gaussian
      for (int j = 0; j < k; j++) {
        P[ik + j] = expf(-distances[ik + j] * beta);
        sum_Pi += P[ik + j];
      }

      // Normalize
      value_t sum_disti_Pi = 0;
      const value_t div    = 1.0f / sum_Pi;
      for (int j = 0; j < k; j++) {
        P[ik + j] *= div;
        sum_disti_Pi += distances[ik + j] * P[ik + j];
      }

      const value_t entropy      = logf(sum_Pi) + beta * sum_disti_Pi;
      const value_t entropy_diff = entropy - desired_entropy;
      if (fabs(entropy_diff) <= tol) {
        break;
      }

      // Bisection search
      if (entropy_diff > 0) {
        beta_min = beta;
        if (isinf(beta_max))
          beta *= 2.0f;
        else
          beta = (beta + beta_max) * 0.5f;
      } else {
        beta_max = beta;
        if (isinf(beta_min))
          beta *= 0.5f;
        else
          beta = (beta + beta_min) * 0.5f;
      }
    }
  }
}
