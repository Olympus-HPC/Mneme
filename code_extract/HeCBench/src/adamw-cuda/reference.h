static const float exp_qmap_ [] = {
  -0.8875, -0.6625, -0.4375, -0.2125, -0.0775, -0.0325, -0.0055, 0.0000, 0.0055, 0.0325, 0.0775, 0.2125, 0.4375, 0.6625, 0.8875, 1.0000,
};

static const float exp_qmidpt_ [] = {
  -0.775, -0.55, -0.325, -0.145, -0.055, -0.019, -0.00275, 0.00275, 0.019, 0.055, 0.145, 0.325, 0.55, 0.775, 0.94375,
};

static const float sq_qmap_ [] = {
  0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 0.4375, 0.5000, 0.5625, 0.6250, 0.6875, 0.7500, 0.8125, 0.8750, 0.9375, 1.0000,
};

static const float sq_qmidpt_ [] = {
  0.09375, 0.15625, 0.21875, 0.28125, 0.34375, 0.40625, 0.46875, 0.53125, 0.59375, 0.65625, 0.71875, 0.78125, 0.84375, 0.90625, 0.96875,
};


template <typename T, int block_size>
void reference (
    const int grid_size,
    T* __restrict__ p,
    const T* __restrict__ g,
    T* __restrict__ exp_qscale,// m
    T* __restrict__ sq_qscale, // v
    int8_t* __restrict__ exp,
    int8_t* __restrict__ sq,
    const float beta1,
    const float beta2,
    const float lr,
    const float weight_decay,
    const float eps,
    const float step,
    const int64_t total_size,
    const float correction1,
    const float correction2_sqrt,
    const float step_size,
    const float weight_decay_update,
    const float resid_beta1,
    const float resid_beta2)
{
  for (int block_id = 0; block_id < grid_size; block_id++) {  
    float absmax_exp = 0;
    float absmax_sq = 0;
    float local_exp_left[block_size];
    float local_sq_left[block_size];
    float local_exp_right[block_size];
    float local_sq_right[block_size];
    for (int thread_id = 0; thread_id < block_size; thread_id++) {
      int64_t global_id = (int64_t)block_id * block_size + thread_id;
      if (global_id >= total_size) break;
      const int8_t exp_full = exp[global_id];
      const int8_t sq_full = sq[global_id];

      float2 p2 = reinterpret_cast<float2*>(p)[global_id];
      const float2 g2 = reinterpret_cast<const float2*>(g)[global_id];

      // left side processing -------------------------------------
      const int8_t exp_left_index = exp_full & _bitmask;
      const int8_t sq_left_index = sq_full & _bitmask;

      //decoupled weight decay
      p2.x = p2.x * weight_decay_update;

      // left exp and sq updates
      float exp_avg_qscale = exp_qscale[block_id];

      float exp_left = exp_qmap_[exp_left_index] * exp_avg_qscale;
      exp_left = beta1 * exp_left + resid_beta1 * g2.x;

      float sq_left = sq_qmap_[sq_left_index] * sq_qscale[block_id];
      sq_left = beta2 * sq_left + resid_beta2 * (g2.x * g2.x);

      // param update
      p[global_id*2] = p2.x - (step_size * (exp_left/(sqrtf(sq_left) / correction2_sqrt + eps)));

      // right side processing -------------------------------

      const int8_t exp_right_index = (exp_full >> 4) & _bitmask;
      const int8_t sq_right_index = (sq_full >> 4) & _bitmask;

      //decoupled weight decay, right side
      p2.y = p2.y * weight_decay_update;

      float exp_right = exp_qmap_[exp_right_index] * exp_avg_qscale;
      exp_right = beta1 * exp_right + resid_beta1 * g2.y;

      float sq_right = sq_qmap_[sq_right_index] * sq_qscale[block_id];
      sq_right = beta2 * sq_right + resid_beta2 * (g2.y * g2.y);

      // param update
      p[global_id*2+1] = p2.y - (step_size * (exp_right/(sqrtf(sq_right) / correction2_sqrt + eps)));

      // prepare quantization info - update absmax scales
      float local_absmax_exp = fmaxf(exp_left, exp_right);
      float local_absmax_sq = fmaxf(sq_left, sq_right);

      local_exp_left[thread_id] = exp_left;
      local_exp_right[thread_id] = exp_right;
      local_sq_left[thread_id] = sq_left;
      local_sq_right[thread_id] = sq_right;

      // determine absmax for exp
      absmax_exp = fmaxf(absmax_exp, local_absmax_exp);

      // same for sq
      absmax_sq = fmaxf(absmax_sq, local_absmax_sq);
    }

    exp_qscale[block_id] = absmax_exp;
    sq_qscale[block_id] = absmax_sq;

    for (int thread_id = 0; thread_id < block_size; thread_id++) {
      int64_t global_id = (int64_t)block_id * block_size + thread_id;
      if (global_id >= total_size) break;

      int8_t local_packed_exp = 0;
      int8_t local_packed_sq = 0;

      // quantize and pack
      const int8_t q_exp_left = (int8_t)q_mapping (exp_qmap_, exp_qmidpt_, (float)local_exp_left[thread_id] / absmax_exp);
      const int8_t q_sq_left = (int8_t)q_mapping (sq_qmap_, sq_qmidpt_, (float)local_sq_left[thread_id] / absmax_sq);
      local_packed_exp |= (q_exp_left & _bitmask);
      local_packed_sq |= (q_sq_left & _bitmask);

      const int8_t q_exp_right = (int8_t)q_mapping (exp_qmap_, exp_qmidpt_, (float)local_exp_right[thread_id] / absmax_exp);
      const int8_t q_sq_right = (int8_t)q_mapping (sq_qmap_, sq_qmidpt_, (float)local_sq_right[thread_id] / absmax_sq);
      local_packed_exp |= (q_exp_right & _right_pack_bitmask);
      local_packed_sq |= (q_sq_right & _right_pack_bitmask);

      // store updated exp and sq
      exp[global_id] = local_packed_exp;
      sq[global_id] = local_packed_sq;
    }
  }
}
