int8_t float2int8(float f, float scale) {
  int8_t i = int8_t(f * scale);
  if (i < -127) i = -127;
  if (i > 127) i = 127;
  return i;
}

void performance (int m, int n, int k, bool is_integer, double avg_time) {
  double total_ops = double(m) * double(n) * double(k) * 2;
  double perf = (total_ops / avg_time) * 1e-9;

  auto scale_string = "G";
  auto unit_string = is_integer ? "OP/s" : "FLOP/s";

  if (perf >= 1000) {
    perf /= 1000;
    scale_string = "T";
  }

  printf("%lf %s%s\n", perf, scale_string, unit_string);
}
