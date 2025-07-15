#if defined(__CUDACC__) || defined(__HIPCC__)
  #define HOST_DEVICE __host__ __device__
#else
  #define HOST_DEVICE
#endif

HOST_DEVICE
float sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * (tgt - (lgt >= 0.f)) - logf(1.f + expf(lgt - 2.f * lgt * (lgt >= 0.f)));
}

HOST_DEVICE
float sigmoid_partition(float lgt) {
  return lgt * (lgt >= 0.f) + logf(1.f + expf(lgt - 2.f * lgt * (lgt >= 0.f)));
}

HOST_DEVICE
float sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt) {
  return (2.f * tgt - 1.f) * (lgt - sigmoid_partition(lgt));
}

HOST_DEVICE
float unjoined_sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * tgt + (tgt - 1.f) * lgt * (lgt >= 0.f) -
      (1.f - tgt) * logf(1.f + expf(lgt - 2.f * lgt * (lgt >= 0.f)));
}

void reference (
  const int outer_size,
  const int inner_size,
  const bool log_D_trick,
  const bool unjoined_lr_loss,
  const float* logits_ptr,
  const float* targets_ptr,
        float* out_ptr)
{
  for (int i = 0; i < outer_size; i++) {
    float value = 0;
    for (int in_idx = i * inner_size;
             in_idx < (i+1) * inner_size; in_idx++) {
      float lgt = logits_ptr[in_idx];
      float tgt = targets_ptr[in_idx];
      if (unjoined_lr_loss) {
        value += unjoined_sigmoid_xent_forward(lgt, tgt);
      } else {
        value += log_D_trick ?
                 sigmoid_xent_forward_with_log_d_trick(lgt, tgt) :
                 sigmoid_xent_forward(lgt, tgt);
      }
    }
    out_ptr[i] = -value / inner_size;
  }
}
