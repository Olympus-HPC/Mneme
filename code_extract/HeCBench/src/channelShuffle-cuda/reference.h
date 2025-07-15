template <typename T>
void ChannelShuffleNCHWKernel_cpu(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
          T* Y)
{
  const int C = G * K;
  #pragma omp parallel for collapse(3)
  for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
      for (int s = 0; s < HxW; s++)
        Y[(n * C + c) * HxW + s] = X[(n * C + (c % G) * K + c / G) * HxW + s];
}

template <typename T, int kSharedSize>
void ChannelShuffleNHWCKernel_cpu(const int O, const int G, const int K, const T* X, T* Y)
{
  const int C = G * K;
  #pragma omp parallel for collapse(2)
  for (int o = 0; o < O; o++)
    for (int i = 0; i < C; i++)
      Y[o*C + i] = X[o*C + (i % G) * K + i / G];
}

template <typename T>
bool ChannelShuffleNCHW_cpu (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;
  const int K = C / G;
  const int HxW = numel / (N * C);
  ChannelShuffleNCHWKernel_cpu<float>(N, G, K, HxW, X, Y);
  return true;
}

template <typename T>
bool ChannelShuffleNHWC_cpu (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int O = N * HxW;

  if (C <= 32) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel_cpu<float, 32>(O, G, K, X, Y);
  } else if (C <= 128) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel_cpu<float, 128>(O, G, K, X, Y);
  } else if (C <= 512) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel_cpu<float, 512>(O, G, K, X, Y);
  }

  return true;
}
