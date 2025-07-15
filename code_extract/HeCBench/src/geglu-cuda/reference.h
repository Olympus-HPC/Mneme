template<typename opmath_t>
opmath_t gelu_reference(opmath_t x) {
    constexpr opmath_t kAlpha = M_SQRT1_2;
    return x * opmath_t(0.5) * (opmath_t(1) + erf(x * kAlpha));
}

template<typename scalar_t>
void geglu_reference(scalar_t *out, const scalar_t *x_and_gate, int n, int dim_last ) {
  for (int i = 0; i < n; i++) {
    for (int d = 0; d < dim_last; d++) {
      scalar_t ux = x_and_gate[(i*2 + 0) * dim_last + d];
      scalar_t ug = x_and_gate[(i*2 + 1) * dim_last + d];
      out[i * dim_last + d] = ux * gelu_reference(ug);
    }
  }
}
