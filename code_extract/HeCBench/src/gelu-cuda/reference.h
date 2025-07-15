void gelu_bias_loop_cpu(__half* src, const __half* bias, int batch_size, int width, int height)
{
  #pragma omp parallel for collapse(3)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int x = 0; x < height; x++) {
      for (int y = 0; y < width; y = y + 2) {
        __half2 v_bias = ((half2*)bias)[y >> 1];
        __half2 v_src  = ((half2*)src)[(batch * width * height + x * width + y) >> 1];
        __half2 v      = __hadd2(v_src, v_bias);
        float2 t     = __half22float2(v);
        t.x    = (0.5f * t.x * (1.0f + tanhf(0.79788456f * (t.x + 0.044715f * t.x * t.x * t.x))));
        t.y    = (0.5f * t.y * (1.0f + tanhf(0.79788456f * (t.y + 0.044715f * t.y * t.y * t.y))));
        ((half2*)src)[(batch * width * height + x * width + y) >> 1] = __float22half2_rn(t);
      }
    }
  }
}

