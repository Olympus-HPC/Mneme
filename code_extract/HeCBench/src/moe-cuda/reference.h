#include <cmath>

void moeSoftmax_reference(
    const float* input,
    const bool* finished,
    float* output,
    const int num_tokens,
    const int num_cols)
{
  for (int token = 0; token < num_tokens; token++) {

    const int thread_row_offset = token * num_cols;

    float threadData(-FLT_MAX);

    if ((finished != nullptr) && finished[token]) {
      return;
    }

    for (int ii = 0; ii < num_cols; ii ++) {
      const int idx = thread_row_offset + ii;
      threadData = fmaxf(static_cast<float>(input[idx]), threadData);
    }

    float float_max = threadData;

    threadData = 0;

    for (int ii = 0; ii < num_cols; ii ++) {
      const int idx = thread_row_offset + ii;
      threadData += expf((static_cast<float>(input[idx]) - float_max));
    }

    const auto Z = threadData;

    float normalizing_factor = 1.f / Z;

    for (int ii = 0; ii < num_cols; ii ++) {
      const int idx = thread_row_offset + ii;
      const float val = expf((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
      output[idx] = val;
    }
  }
}

struct kvp { // KeyValuePair
  int key;
  float value; 
};

kvp arg_max(const kvp &a, const kvp &b) {
  if (a.value > b.value)
    return {a.key, a.value};
  else if (a.value == b.value)
    return {std::min(a.key, b.key), a.value};
  else
    return {b.key, b.value};
}

void moeTopK_reference(
    const float* inputs_after_softmax,
    const bool* finished,
    float* output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_tokens,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{
  kvp thread_kvp;

  for (int token = 0; token < num_tokens; token++) {
    const bool row_is_active = finished ? !finished[token] : true;
    const int thread_read_offset = token * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      thread_kvp.key = 0;
      thread_kvp.value = -1.f;  // This is OK because inputs are probabilities

      kvp inp_kvp;
      for (int expert = 0; expert < num_experts; expert++) {
        const int idx = thread_read_offset + expert;
        inp_kvp.key = expert;
        inp_kvp.value = inputs_after_softmax[idx];

        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
          const int prior_winning_expert = indices[k * token + prior_k];
          if (prior_winning_expert == expert) {
            inp_kvp = thread_kvp;
          }
        }
        thread_kvp = arg_max(inp_kvp, thread_kvp);
      }

      const int expert = thread_kvp.key;
      const bool node_uses_expert = expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * token + k_idx;
      output[idx] = thread_kvp.value;
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      source_rows[idx] = k_idx * num_tokens + token;
    }
  }
}
