using namespace std;

// generate rand int64_t [a, b]
#define random_int(a, b) ( rand() % (b - a) + a )

// generate rand float [0, 1]
#define random_float() (rand() / double(RAND_MAX))

// This version fused the log_softmax
#define tolerance 4e-3

// tunable thread block size
#define threadX  64
#define threadBS 1

int64_t errors(0);
constexpr int bs = 128;
constexpr int W = 81;
constexpr int H = 8732;
constexpr int PredictShape = bs * W * H;
constexpr int TargetShape = bs * H;
constexpr int OutputShape = bs * H;

__host__ __half f2h (float x) { return __float2half(x); }

__inline__ __host__ __device__
float h2f (__half x) { return __half2float(x); }

template <typename scalar_t, typename gscalar_t>
void loss_bwd_cpu(scalar_t* predict, int64_t* target, scalar_t* weight, int64_t* mask,
                  gscalar_t* grad_output, gscalar_t* grad_output_neg, gscalar_t* grad_predict)
{
  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = i * H + j;
        int64_t idx = target[offset];
        int64_t predict_offset = i * W * H + k * H + j;

        if (idx == int64_t(k)) {
          grad_predict[predict_offset] = (-grad_output[offset] * weight[offset]) +
            (mask[offset] ? -grad_output_neg[offset] * weight[offset] : 0);
        }
        else {
          grad_predict[predict_offset] = 0;
        }
      }
    }
  }

  vector<vector<float>> sum_value;
  for (int i = 0; i < bs; ++i) {
    vector<float> bs_sum_value;
    for (int j = 0; j < H; ++j) {
      float sum = 0.0;
      for (int k = 0; k < W; ++k) {
        int64_t offset = i * W * H + k * H + j;
        sum += grad_predict[offset] * predict[offset];
      }
      bs_sum_value.push_back(sum);
    }
    sum_value.push_back(bs_sum_value);
  }

  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = i * W * H + k * H + j;
        grad_predict[offset] = grad_predict[offset] - std::exp(predict[offset]) * sum_value[i][j];
      }
    }
  }
}

template <>
void loss_bwd_cpu<__half, __half>(__half* predict, int64_t* target, __half* weight, int64_t* mask,
                  __half* grad_output, __half* grad_output_neg, __half* grad_predict)
{
  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = i * H + j;
        int64_t idx = target[offset];
        int64_t predict_offset = i * W * H + k * H + j;

        if (idx == int64_t(k)) {
          grad_predict[predict_offset] = f2h((-h2f(grad_output[offset]) * h2f(weight[offset])) +
            (mask[offset] ? -h2f(grad_output_neg[offset]) * h2f(weight[offset]) : 0));
        }
        else {
          grad_predict[predict_offset] = 0;
        }
      }
    }
  }

  vector<vector<float>> sum_value;
  for (int i = 0; i < bs; ++i) {
    vector<float> bs_sum_value;
    for (int j = 0; j < H; ++j) {
      float sum = 0.0;
      for (int k = 0; k < W; ++k) {
        int64_t offset = i * W * H + k * H + j;
        sum += h2f(grad_predict[offset]) * h2f(predict[offset]);
      }
      bs_sum_value.push_back(sum);
    }
    sum_value.push_back(bs_sum_value);
  }

  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = i * W * H + k * H + j;
        grad_predict[offset] = f2h(h2f(grad_predict[offset]) - std::exp(h2f(predict[offset])) * sum_value[i][j]);
      }
    }
  }
}

template <typename scalar_t>
void verify(scalar_t* output, scalar_t* output_device, size_t sz) {
  int count = 0;
  for (size_t i = 0; i < sz; ++i) {
    int64_t offset = i;
    if (std::abs(output[offset] - output_device[offset]) > tolerance) {
      count++;
      if (count < 10) 
        std::cout << "Error, output not equal, i="
                  << i << ", cpu_result = " << output[offset] << ", device_result = "
                  << output_device[offset] << ", gap = "
                  << output[offset] - output_device[offset] << std::endl;
    }
  }
  errors += count;
}

template <>
void verify<__half>(__half* output, __half* output_device, size_t sz) {
  int count = 0;
  for (size_t i = 0; i < sz; ++i) {
    int64_t offset = i;
    if (std::abs(h2f(output[offset]) - h2f(output_device[offset])) > tolerance) {
      count++;
      if (count < 10) 
        std::cout << "Error, output not equal, i="
                  << i << ", cpu_result = " << h2f(output[offset]) << ", device_result = "
                  << h2f(output_device[offset]) << ", gap = "
                  << h2f(output[offset]) - h2f(output_device[offset]) << std::endl;
    }
  }
  errors += count;
}
