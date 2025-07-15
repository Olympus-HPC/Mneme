#include <algorithm> // max/min

template <typename PoolProcess, typename T>
void reference(
    const int nthreads,
    const T*__restrict input_data, 
    const T*__restrict output_data,
    const T*__restrict output_grad, 
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int ksize_height,
    const int ksize_width,
    const int stride_height,
    const int stride_width,
    const int padding_height,
    const int padding_width,
    PoolProcess pool_process, 
    bool exclusive,
    T*__restrict input_grad,
    bool channel_last = false)
{
  for (int index = 0; index < nthreads; index ++) {
    int w_offset, h_offset, offsetC, batch_idx;
    if (!channel_last) { /* NCHW */
      w_offset = index % input_width + padding_width;
      h_offset = (index / input_width) % input_height + padding_height;
      offsetC = (index / input_width / input_height) % channels;
      batch_idx = index / input_width / input_height / channels;
    } else { /* NHWC */
      offsetC = index % channels;
      w_offset = (index / channels) % input_width + padding_width;
      h_offset =
        (index / channels / input_width) % input_height + padding_height;
      batch_idx = index / channels / input_width / input_height;
    }

    int phstart = (h_offset < ksize_height) ? 0 : (h_offset - ksize_height) / stride_height + 1;
    int pwstart = (w_offset < ksize_width) ? 0 : (w_offset - ksize_width) / stride_width + 1;
    int phend = std::min(h_offset / stride_height + 1, output_height);
    int pwend = std::min(w_offset / stride_width + 1, output_width);

    // initial gradient value
    T gradient = static_cast<T>(0.0);
    T input = input_data[index];

    int output_stride;
    if (!channel_last) {
      output_stride = (batch_idx * channels + offsetC) * output_height * output_width;
    } else {
      output_stride = batch_idx * output_height * output_width * channels;
    }

    const T *output_data_t = output_data + output_stride;
    const T *output_grad_t = output_grad + output_stride;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int pool_size;
        int hstart = ph * stride_height - padding_height;
        int wstart = pw * stride_width - padding_width;
        int hend = std::min(hstart + ksize_height, input_height);
        int wend = std::min(wstart + ksize_width, input_width);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        pool_size = exclusive ? (hend - hstart) * (wend - wstart)
          : ksize_height * ksize_width;

        int output_sub_idx = channel_last
          ? (ph * output_width + pw) * channels + offsetC
          : ph * output_width + pw;
        pool_process.compute(input, output_data_t[output_sub_idx],
            output_grad_t[output_sub_idx],
            static_cast<T>(1.f / pool_size), &gradient);
      }
    }
    input_grad[index] = gradient;
  }
}

