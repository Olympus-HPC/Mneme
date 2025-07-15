#ifndef KERNELS
#define KERNELS

__global__
void lrn_fwd_kernel(
    const float* __restrict__ src_, 
          float* __restrict__ dst_,
    int64_t N_,
    int64_t C_, 
    int64_t D_, 
    int64_t H_, 
    int64_t W_, 
    int64_t stride_mb_, 
    int64_t ndims_, 
    int64_t wk_size_, 
    int64_t size_, 
    float alpha_, 
    float beta_, 
    float k_)
{
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
               idx < wk_size_ ; idx += blockDim.x * gridDim.x) {

    auto data_off = [=](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
      int64_t tag = 0;
      switch (tag) {
        case 0 : return mb * stride_mb_ + c * H_ * W_ + h * W_ + w;
        case 1 : return mb * stride_mb_ + h * W_ * C_ + w * C_ + c;
        default:
           return (int64_t)1;
      }
    };

    auto ker = [=](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) {
      float sum = 0;
      const int64_t half_size = (size_ - 1) / 2;
      bool across_channel = 1;
      if (across_channel) {
        const int64_t c_st = max(oc - half_size + 0, (int64_t)0);
        const int64_t c_en = min(oc + half_size + 1, C_);

        for (int64_t c = c_st; c < c_en; ++c) {
          const auto s_off = data_off(mb, c, od, oh, ow);
          const auto s = src_[s_off];
          sum+=s*s;
        }
      } else {
        int64_t d_st = max(od - half_size + 0, (int64_t)0);
        int64_t d_en = min(od + half_size + 1, D_);
        int64_t h_st = max(oh - half_size + 0, (int64_t)0);
        int64_t h_en = min(oh + half_size + 1, H_);
        int64_t w_st = max(ow - half_size + 0, (int64_t)0);
        int64_t w_en = min(ow + half_size + 1, W_);
        for (int64_t d = d_st; d < d_en; ++d) {
          for (int64_t h = h_st; h < h_en; ++h) {
            for (int64_t w = w_st; w < w_en; ++w) {
              const auto s_off = data_off(mb, oc, d, h, w);
              const auto s = src_[s_off];
              sum+=s*s;
            }
          }
        }
      }
      sum = k_ + alpha_ * sum / size_;
      const auto s_off = data_off(mb, oc, od, oh, ow);
      const auto s = src_[s_off];
      return (s * sqrtf(1.0f / (sqrtf(sum) * sum)));
    };

    auto Operation = [=]( int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
      bool channel = 0;
      if(channel) {
        const int64_t off = mb * stride_mb_ + h * W_ * C_ + w * C_ + c;
        auto val = ker(mb, c, 0, h, w);
        dst_[off] = val;
      }   
      else {
        const int64_t off = data_off(mb, c, d, h, w);
        auto val = ker(mb, c, d, h, w);
        dst_[off] = val;
      }
    };

    int64_t n = (idx / (C_ * D_ * H_ * W_)) % N_;
    int64_t c = (idx / (D_ * H_ * W_)) % C_;
    int64_t d = (idx / (H_ * W_)) % D_;
    int64_t h = (idx / (W_)) % H_;
    int64_t w = (idx / (1)) % W_;

    Operation(n, c, d, h, w);
  }
}

__global__
void  lrn_bwd_kernel(
    const float* __restrict__ src_, 
          float* __restrict__ dst_,
          float* __restrict__ diff_src_mem_, 
    int64_t N_,
    int64_t C_, 
    int64_t D_, 
    int64_t H_, 
    int64_t W_, 
    int64_t stride_mb_, 
    int64_t ndims_, 
    int64_t wk_size_, 
    int64_t size_, 
    float alpha_, 
    float beta_, 
    float k_)
{
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
               idx < wk_size_; idx += blockDim.x * gridDim.x) {

    auto data_off = [=](int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
      int64_t tag = 0;
      switch (tag) {
        case 0 : return mb * stride_mb_ + c * H_ * W_ + h * W_ + w;
        case 1 : return mb * stride_mb_ + h * W_ * C_ + w * C_ + c;
        default:
           return (int64_t)1;
      }
    };

    auto get_omega = [=](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) {
      auto sum = 0;
      const int64_t half_size = (size_ - 1) / 2;
      bool across_channel = 1;
      if (across_channel) {
        const int64_t c_st = max(oc - half_size + 0, (int64_t)0);
        const int64_t c_en = min(oc + half_size + 1, C_);

        for (int64_t c = c_st; c < c_en; ++c) {
          const auto s_off = data_off(mb, c, od, oh, ow);
          const auto s = src_[s_off];
          sum += s * s;
        }
      } else {
        int64_t d_st = max(od - half_size + 0, (int64_t)0);
        int64_t d_en = min(od + half_size + 1, D_);
        int64_t h_st = max(oh - half_size + 0, (int64_t)0);
        int64_t h_en = min(oh + half_size + 1, H_);
        int64_t w_st = max(ow - half_size + 0, (int64_t)0);
        int64_t w_en = min(ow + half_size + 1, W_);
        for (int64_t d = d_st; d < d_en; ++d)
          for (int64_t h = h_st; h < h_en; ++h)
            for (int64_t w = w_st; w < w_en; ++w) {
              const auto s_off = data_off(mb, oc, d, h, w);
              const auto s = src_[s_off];
              sum += s * s;
            }
      }
      return (k_ + alpha_ * sum / size_);
    };

    auto ker = [=](int64_t mb, int64_t oc, int64_t od, int64_t oh, int64_t ow) {
      float A = 0, B = 0;
      const int64_t half_size = (size_ - 1) / 2;
      bool across_channel = 1;
      if (across_channel) {
        const int64_t c_st = max(oc - half_size + 0, (int64_t)0);
        const int64_t c_en = min(oc + half_size + 1, C_);

        for (int64_t c = c_st; c < c_en; ++c) {
          const auto off = data_off(mb, c, od, oh, ow);
          const auto omega = get_omega(mb, c, od, oh, ow);
          const auto omega_in_beta
            = sqrtf(1.0f / (sqrtf(omega) * omega));

          const auto dst_val = dst_[off];
          const auto tmp = omega_in_beta * dst_val;
          if (c == oc) A = tmp;
          const auto src_val = src_[off];
          B += (src_val * tmp / omega);
        }
      } else {
        int64_t d_st = max(od - half_size + 0, (int64_t)0);
        int64_t d_en = min(od + half_size + 1, D_);
        int64_t h_st = max(oh - half_size + 0, (int64_t)0);
        int64_t h_en = min(oh + half_size + 1, H_);
        int64_t w_st = max(ow - half_size + 0, (int64_t)0);
        int64_t w_en = min(ow + half_size + 1, W_);
        for (int64_t d = d_st; d < d_en; ++d)
          for (int64_t h = h_st; h < h_en; ++h)
            for (int64_t w = w_st; w < w_en; ++w) {
              const auto off = data_off(mb, oc, d, h, w);
              const auto omega = get_omega(mb, oc, d, h, w);
              const auto omega_in_beta
                = sqrtf(1.0f / (sqrtf(omega) * omega));

              const auto dst_val = dst_[off];
              const auto tmp
                = omega_in_beta * dst_val;
              if (d == od && h == oh && w == ow) A = tmp;
              const auto src_val = src_[off];
              B += (src_val * tmp / omega);
            }
      }
      const auto off = data_off(mb, oc, od, oh, ow);
      const auto src_val = src_[off];
      B *= (2.0f * alpha_ * beta_ * src_val / size_);
      return (A - B);
    };

    auto Operation = [=]( int64_t mb, int64_t c, int64_t d, int64_t h, int64_t w) {
      bool channel = 0;
      if(channel) {
        const int64_t off = mb * stride_mb_ + h * W_ * C_ + w * C_ + c;
        auto val = ker(mb, c, 0, h, w);
        dst_[off] = val;
      }   
      else {
        const int64_t off = data_off(mb, c, d, h, w);
        auto val = ker(mb, c, d, h, w);
        diff_src_mem_[off] = val;
      }
    };

    auto n = (idx / (C_ * D_ * H_ * W_)) % N_;
    auto c = (idx / (D_ * H_ * W_)) % C_;
    auto d = (idx / (H_ * W_)) % D_;
    auto h = (idx / (W_)) % H_;
    auto w = (idx / (1)) % W_;

    Operation(n, c, d, h, w);
  }
}
#endif
