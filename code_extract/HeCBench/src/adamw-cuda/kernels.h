#define HOST_DEVICE __host__ __device__
#define DEVICE __device__

#include <cub/cub.cuh>

template <typename T, int block_size>
__device__ void BlockReduce(T &input1, T &input2) {
  using BlockReduce = cub::BlockReduce<T, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;
  input1 = BlockReduce(temp_storage1).Reduce(input1, cub::Max());
  input2 = BlockReduce(temp_storage2).Reduce(input2, cub::Max());
}

static DEVICE __const__ uint8_t _bitmask = 15;
static DEVICE __const__ uint8_t _right_pack_bitmask = _bitmask << 4;

static DEVICE __const__ float _exp_qmap [] = {
                -0.8875,
                -0.6625,
                -0.4375,
                -0.2125,
                -0.0775,
                -0.0325,
                -0.0055,
                0.0000,
                0.0055,
                0.0325,
                0.0775,
                0.2125,
                0.4375,
                0.6625,
                0.8875,
                1.0000,
};

static DEVICE __const__ float _exp_qmidpt [] = {

            -0.775,
            -0.55,
            -0.325,
            -0.145,
            -0.055,
            -0.019,
            -0.00275,
            0.00275,
            0.019,
            0.055,
            0.145,
            0.325,
            0.55,
            0.775,
            0.94375,
};

static DEVICE __const__ float _sq_qmap [] = {
                0.0625,
                0.1250,
                0.1875,
                0.2500,
                0.3125,
                0.3750,
                0.4375,
                0.5000,
                0.5625,
                0.6250,
                0.6875,
                0.7500,
                0.8125,
                0.8750,
                0.9375,
                1.0000,
};

static DEVICE __const__ float _sq_qmidpt [] = {
            0.09375,
            0.15625,
            0.21875,
            0.28125,
            0.34375,
            0.40625,
            0.46875,
            0.53125,
            0.59375,
            0.65625,
            0.71875,
            0.78125,
            0.84375,
            0.90625,
            0.96875,
};

// binary search for quantization
HOST_DEVICE __forceinline__ float q_mapping(const float* __restrict__ qmap,
                                            const float* __restrict__ qmidpt,
                                            float x)
{
    // 4 bit range
    int low = 0;
    int high = 15;

    if (x <= qmap[low]) return low;
    if (qmap[high] <=x) return high;

    // replace with for loop?
    while (low < high) {
        int mid = (low + high) >> 1;
        if (qmap[mid] <= x)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
    }
    return (qmidpt[low-1] < x) ? low : low-1;
}

template <typename T, int block_size = 64>
__global__ void fused_4bit_kernel(
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
    __shared__ float absmax_exp;
    __shared__ float absmax_sq;

    float local_absmax_exp = 0;
    float local_absmax_sq = 0;
    int8_t local_packed_exp = 0;
    int8_t local_packed_sq = 0;

    int64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < total_size) {

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
      float exp_avg_qscale = exp_qscale[blockIdx.x];

      float exp_left = _exp_qmap[exp_left_index] * exp_avg_qscale;
      exp_left = beta1 * exp_left + resid_beta1 * g2.x;

      float sq_left = _sq_qmap[sq_left_index] * sq_qscale[blockIdx.x];
      sq_left = beta2 * sq_left + resid_beta2 * (g2.x * g2.x);

      // param update
      p[global_id*2] = p2.x - (step_size * (exp_left/(sqrtf(sq_left) / correction2_sqrt + eps)));

      // right side processing -------------------------------

      const int8_t exp_right_index = (exp_full >> 4) & _bitmask;
      const int8_t sq_right_index = (sq_full >> 4) & _bitmask;

      //decoupled weight decay, right side
      p2.y = p2.y * weight_decay_update;

      float exp_right = _exp_qmap[exp_right_index] * exp_avg_qscale;
      exp_right = beta1 * exp_right + resid_beta1 * g2.y;

      float sq_right = _sq_qmap[sq_right_index] * sq_qscale[blockIdx.x];
      sq_right = beta2 * sq_right + resid_beta2 * (g2.y * g2.y);

      // param update
      p[global_id*2+1] = p2.y - (step_size * (exp_right/(sqrtf(sq_right) / correction2_sqrt + eps)));

      // prepare quantization info - update absmax scales
      local_absmax_exp = fmaxf(exp_left, exp_right);
      local_absmax_sq = fmaxf(sq_left, sq_right);

      // parallel reduction to determine absmax for exp and sq
      BlockReduce<T, block_size>(local_absmax_exp, local_absmax_sq);
      if (threadIdx.x ==0) {
          exp_qscale[blockIdx.x] = local_absmax_exp;
          sq_qscale[blockIdx.x] = local_absmax_sq;
          absmax_exp = local_absmax_exp;
          absmax_sq = local_absmax_sq;
      }
      __syncthreads();

      // quantize and pack
      const int8_t q_exp_left = (int8_t)q_mapping(_exp_qmap, _exp_qmidpt, exp_left / absmax_exp);
      const int8_t q_sq_left = (int8_t)q_mapping(_sq_qmap, _sq_qmidpt, sq_left / absmax_sq);
      local_packed_exp |= (q_exp_left & _bitmask);
      local_packed_sq |= (q_sq_left & _bitmask);

      const int8_t q_exp_right = (int8_t)q_mapping(_exp_qmap, _exp_qmidpt, exp_right / absmax_exp);
      const int8_t q_sq_right = (int8_t)q_mapping(_sq_qmap, _sq_qmidpt, sq_right / absmax_sq);
      local_packed_exp |= (q_exp_right & _right_pack_bitmask);
      local_packed_sq |= (q_sq_right & _right_pack_bitmask);

      // store updated exp and sq
      exp[global_id] = local_packed_exp;
      sq[global_id] = local_packed_sq;
   }
}
