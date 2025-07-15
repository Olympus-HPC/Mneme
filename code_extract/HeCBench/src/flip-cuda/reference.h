template <typename scalar_t>
void flip_kernel_cpu(
    const scalar_t* in_tensor,
          scalar_t* out_tensor,
    int64_t  n,
    const int64_t* flip_dims,
    const int64_t  flip_dims_size,
    const int64_t* strides,
    const int64_t* strides_contiguous,
    const int64_t* shape,
    const int64_t  total_dims) 
{
  #pragma omp parallel for
  for (int64_t lid = 0; lid < n; lid++) {
    int64_t cur_indices = lid;
    int64_t rem = 0;
    int64_t dst_offset = 0;

    for (int64_t i = 0; i < total_dims; i++) {
      int64_t temp = cur_indices;
      cur_indices = cur_indices / strides_contiguous[i];
      rem = temp - cur_indices * strides_contiguous[i];
      for (int64_t j = 0; j < flip_dims_size; j++) {
        // flip the indices if it is in flip_dims
        if (i == flip_dims[j]) {
          cur_indices = shape[i] - 1 - cur_indices;
        }
      }
      dst_offset += cur_indices * strides[i];
      cur_indices = rem;
    }
    out_tensor[lid] = in_tensor[dst_offset];
  }
}

