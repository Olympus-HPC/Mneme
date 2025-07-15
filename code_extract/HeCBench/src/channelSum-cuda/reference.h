template <typename T>
void ref_nchw (
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  for (int c = 0; c < C; c++) {
    T m_val = 0, v_val = 0;
    for (int n = 0; n < N; n++) {
      for (int hw = 0; hw < HxW; hw++) {
        const int index = (n * C + c) * HxW + hw;
        m_val += *(X + index);
        v_val += *(X + index) * *(X + index);
      }
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
void ref_nhwc (
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  for (int c = 0; c < C; c++) {
    T m_val = 0, v_val = 0;
    for (int i = 0; i < N * HxW; i++) {
        const int index = (i * C + c);
        m_val += *(X + index);
        v_val += *(X + index) * *(X + index);
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}


template <typename T>
bool check (int size, T *d, T *h) {
  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (abs(d[i] - h[i]) > 1) {
      ok = false;
      break;
    }
  }
  return ok;
}
