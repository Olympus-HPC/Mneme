template <typename T>
void vol2col_reference(
    const T* data_vol,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int ksize_t,
    const int ksize_h,
    const int ksize_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    T* data_col)
{
  for (int channel_in = 0; channel_in < channels; channel_in++) {
  for (int t_out = 0; t_out < depth_col; t_out++) {
  for (int h_out = 0; h_out < height_col; h_out++) {
  for (int w_out = 0; w_out < width_col; w_out++) {
    int channel_out = channel_in * ksize_t * ksize_h * ksize_w;
    int t_in = t_out * stride_t - pad_t;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    auto v = data_vol + ((channel_in * depth + t_in) * height + h_in) * width + w_in;
    auto c = data_col + ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out;

    for (int i = 0; i < ksize_t; ++i) {
      for (int j = 0; j < ksize_h; ++j) {
        for (int k = 0; k < ksize_w; ++k) {
          int t = t_in + i * dilation_t;
          int h = h_in + j * dilation_h;
          int w = w_in + k * dilation_w;
          *c = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width)
              ? v[i * dilation_t * height * width +
                  j * dilation_h * width + k * dilation_w]
              : static_cast<T>(0);
          c += depth_col * height_col * width_col;
        }
      }
    }
  } } } }
}

template <typename T, typename accT>
void col2vol_reference(
    const T* data_col,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int pad_t,
    const int pad_h,
    const int pad_w,
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    T* data_vol)
{
  for (int channel_in = 0; channel_in < channels; channel_in++) {
  for (int t_out = 0; t_out < depth; t_out++) {
  for (int h_out = 0; h_out < height; h_out++) {
  for (int w_out = 0; w_out < width; w_out++) {
    accT val = static_cast<accT>(0);
    const int w_im = w_out + pad_w;
    const int h_im = h_out+ pad_h;
    const int t_im = t_out + pad_t;
    const int c_im = channel_in;
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    int kernel_extent_t = (kernel_t - 1) * dilation_t + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = std::min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = std::min(h_im / stride_h + 1, height_col);
    const int t_col_start =
        (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
    const int t_col_end = std::min(t_im / stride_t + 1, depth_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int t_col = t_col_start; t_col < t_col_end; t_col += 1) {
      for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          int64_t t_k = (t_im - t_col * stride_t);
          int64_t h_k = (h_im - h_col * stride_h);
          int64_t w_k = (w_im - w_col * stride_w);
          if (t_k % dilation_t == 0 && h_k % dilation_h == 0 &&
              w_k % dilation_w == 0) {
            t_k /= dilation_t;
            h_k /= dilation_h;
            w_k /= dilation_w;
            const int64_t idx_k =
                ((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k;
            const int64_t data_col_index =
                ((idx_k * depth_col + t_col) *
                    height_col + h_col) *
                  width_col + w_col;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_vol[(int64_t)c_im * width * height * depth + t_out * width * height +
             h_out * width + w_out] = static_cast<T>(val);
  } } } }
}
