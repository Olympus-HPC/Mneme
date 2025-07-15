template <typename T>
void kernel1(T *p_real, T *p_imag, float a, float b, int width, int height) {
  for (int x = 0, peer = width; x < width; x += 2, peer += 2) {
    T tmp_real = p_real[x];
    T tmp_imag = p_imag[x];
    p_real[x] = a * tmp_real - b * p_imag[peer];
    p_imag[x] = a * tmp_imag + b * p_real[peer];
    p_real[peer] = a * p_real[peer] - b * tmp_imag;
    p_imag[peer] = a * p_imag[peer] + b * tmp_real;
  }
  for (int y = 1; y < height - 1; y++) {
    for (int idx = y * width + y % 2, peer = idx + width; idx < (y + 1) * width; idx += 2, peer += 2) {
      T tmp_real = p_real[idx];
      T tmp_imag = p_imag[idx];
      p_real[idx] = a * p_real[idx] - b * p_imag[peer];
      p_imag[idx] = a * p_imag[idx] + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * tmp_imag;
      p_imag[peer] = a * p_imag[peer] + b * tmp_real;
    }
  }
}

template <typename T>
void kernel2(T *p_real, T *p_imag, float a, float b, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int idx = y * width + y % 2, peer = idx + 1; idx < (y + 1) * width - 1; idx += 2, peer += 2) {
      T tmp_real = p_real[idx];
      T tmp_imag = p_imag[idx];
      p_real[idx] = a * tmp_real - b * p_imag[peer];
      p_imag[idx] = a * tmp_imag + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * tmp_imag;
      p_imag[peer] = a * p_imag[peer] + b * tmp_real;
    }
  }
}

template <typename T>
void kernel3(T *p_real, T *p_imag, float a, float b, int width, int height) {
  for (int x = 1, peer = width + 1; x < width; x += 2, peer += 2) {
    float tmp_real = p_real[x];
    float tmp_imag = p_imag[x];
    p_real[x] = a * tmp_real - b * p_imag[peer];
    p_imag[x] = a * tmp_imag + b * p_real[peer];
    p_real[peer] = a * p_real[peer] - b * tmp_imag;
    p_imag[peer] = a * p_imag[peer] + b * tmp_real;
  }
  for (int y = 1; y < height - 1; y++) {
    for (int idx = y * width + 1 - y % 2, peer = idx + width; idx < (y + 1) * width; idx += 2, peer += 2) {
      float tmp_real = p_real[idx];
      float tmp_imag = p_imag[idx];
      p_real[idx] = a * tmp_real - b * p_imag[peer];
      p_imag[idx] = a * tmp_imag + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * tmp_imag;
      p_imag[peer] = a * p_imag[peer] + b * tmp_real;
    }
  }
}

template <typename T>
void kernel4(T *p_real, T *p_imag, float a, float b, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int idx = y * width + 1 - (y % 2), peer = idx + 1; idx < (y + 1) * width - 1; idx += 2, peer += 2) {
      T tmp_real = p_real[idx];
      T tmp_imag = p_imag[idx];
      p_real[idx] = a * tmp_real - b * p_imag[peer];
      p_imag[idx] = a * tmp_imag + b * p_real[peer];
      p_real[peer] = a * p_real[peer] - b * tmp_imag;
      p_imag[peer] = a * p_imag[peer] + b * tmp_real;
    }
  }
}

template <typename T>
void reference(T *p_real, T *p_imag, float a, float b, int width, int height, int repeat) {
  for (int i = 0; i < repeat; i++) {
    kernel1(p_real, p_imag, a, b, width, height);
    kernel2(p_real, p_imag, a, b, width, height);
    kernel3(p_real, p_imag, a, b, width, height);
    kernel4(p_real, p_imag, a, b, width, height);
    kernel4(p_real, p_imag, a, b, width, height);
    kernel3(p_real, p_imag, a, b, width, height);
    kernel2(p_real, p_imag, a, b, width, height);
    kernel1(p_real, p_imag, a, b, width, height);
  }
}
