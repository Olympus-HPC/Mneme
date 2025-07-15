static const float HALF_FLT_MAX = 65504.F;

template<typename T>
void log_probs_cpu (
    float*       log_probs,
    const T*     logits,
    const int*   ids,
    const int*   lengths,
    const int    max_input_length,
    const int    batch_size,
    const int    vocab_size,
    const int    vocab_size_padded,
    float* cum_log_probs)
{
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  for (int bidx = 0; bidx < batch_size; bidx++) {
    float accum = 0.f;
    #pragma omp parallel for reduction(+:accum)
    for (int step = 0; step < lengths[bidx] - 1; step++) {
      int step_offset  = step * vocab_size_padded;
      int batch_offset = bidx * max_input_length * vocab_size_padded;
      auto logits_ptr = logits + step_offset + batch_offset;
      float max_val = -MAX_T_VAL;
      float val;
      #pragma omp parallel for reduction(max:max_val)
      for (int i = 0; i < vocab_size; i++) {
        val = static_cast<float>(logits_ptr[i]);
        max_val = fmaxf(max_val, val);
      }
      float sum_exp = 0.0f;
      #pragma omp parallel for reduction(+:sum_exp)
      for (int i = 0; i < vocab_size; i++) {
        val = expf(static_cast<float>(logits_ptr[i]) - max_val);
        sum_exp += val;
      }
      int idx = step + bidx * (max_input_length - 1);
      // log_probs[step, ...] is the log probability of a token at step t + 1.
      int token_idx = step + 1 + bidx * max_input_length;
      log_probs[idx] = static_cast<float>(logits_ptr[ids[token_idx]]) - max_val - logf(sum_exp + 1e-9f);
      accum += log_probs[idx];
    }
    cum_log_probs[bidx] = accum;
  }
}
