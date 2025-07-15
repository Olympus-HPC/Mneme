
void mha_reference (
   const float *__restrict__ q,
   const float *__restrict__ k,
   const float *__restrict__ v,
   const int beam_size,
   const int n_steps,
   const int qk_col,
   const int v_col,
   const int nhead,
   const float scale,
   const int THRESHOLD,
   float *dst)
{
  int i, t;
  int dim_per_head = qk_col / nhead;
   #pragma omp parallel for collapse(2)
  for (int candidate_id = 0; candidate_id < beam_size; candidate_id++) {
    for (int head_id = 0; head_id < nhead; head_id++) {
      float buffer[qk_col / nhead + n_steps];
      float *sq = buffer;
      float *logits = buffer + dim_per_head;
      #pragma omp parallel for
      for (t = 0; t < dim_per_head; t++) {
        int pos = candidate_id * qk_col + head_id * dim_per_head + t;
        sq[t] = q[pos];
      }
      float dp[n_steps];
      #pragma omp parallel for
      for (t = 0; t < n_steps; t++) {
        float summ = 0.f;
        const float *k2 = k + candidate_id * qk_col * n_steps + head_id * dim_per_head + t * qk_col;
        #pragma omp parallel for reduction(+:summ)
        for (i = 0; i < dim_per_head; i++)
          summ += sq[i] * k2[i];
        summ *= scale;
        dp[t] = summ;
      }
      float max_val = -1e-20f;
      #pragma omp parallel for reduction(max:max_val)
      for (t = 0; t < n_steps; t++) {
        max_val = fmaxf(max_val, dp[t]);
      }
      float val = 0;
      #pragma omp parallel for reduction(+:val)
      for (t = 0; t < n_steps; t++) {
        dp[t] -= max_val;
        if(dp[t] < -THRESHOLD) dp[t] = -THRESHOLD;
        val += expf(dp[t]);
      }
      #pragma omp parallel for
      for (t = 0; t < n_steps; t++) {
        logits[t] = expf(dp[t]) / val;
      }
      #pragma omp parallel for
      for (t = 0; t < dim_per_head; t++) {
        float summ = 0.f;
        int tid = candidate_id * v_col * n_steps + head_id * dim_per_head + t;
        #pragma omp parallel for reduction(+:summ)
        for (i = 0; i < n_steps; ++i)
          summ += logits[i] * v[tid + i * v_col];
        dst[candidate_id * v_col + head_id * dim_per_head + t] = summ;
      }
    }
  }
}
