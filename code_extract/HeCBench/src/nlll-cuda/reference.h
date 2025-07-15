template <typename scalar_t, typename accscalar_t, typename index_t>
void reference(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t*  target,
    const scalar_t* weights,
    bool size_average,
    int64_t nframe,
    int64_t kdim,
    int64_t ignore_index)
{
  accscalar_t output_acc = 0;
  accscalar_t total_weight_acc = 0;
  for (int i = 0; i < nframe; i++) {
    index_t t = target[i];
    if (t != ignore_index) {
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      output_acc -= static_cast<accscalar_t>(input[i * kdim + t] * cur_weight);
      total_weight_acc += static_cast<accscalar_t>(cur_weight);
    }
  }
  *total_weight = static_cast<scalar_t>(total_weight_acc);
  if (size_average) {
    *output = static_cast<scalar_t>(output_acc / total_weight_acc);
  } else {
    *output = static_cast<scalar_t>(output_acc);
  }
}


