void ReluGrad_reference(int count,
                        const half* gradient,
                        const half* feature,
                              half* backprop)
{
  for (int i = 0; i < count; i++) {
    half grad_h = gradient[i];
    half feature_h = feature[i];

    float grad_f = static_cast<float>(grad_h);
    float feature_f = static_cast<float>(feature_h);
    float backprop_f = (feature_f > 0) ? grad_f : 0;

    half backprop_h(backprop_f);
    backprop[i] = backprop_h;
  }
}

void Relu_reference(int count, const int* input, int* output)
{
  for (int i = 0; i < count; i++) {
    signed char c1, c2, c3, c4; 
    unsigned x, y, z, w;
    c1 = input[i] & 0xFF;
    c2 = (input[i] >> 8) & 0xFF;
    c3 = (input[i] >> 16) & 0xFF;
    c4 = (input[i] >> 24) & 0xFF;
    x = c1 > 0 ? c1 : 0;
    y = c2 > 0 ? c2 : 0;
    z = c3 > 0 ? c3 : 0;
    w = c4 > 0 ? c4 : 0;
    output[i] = w << 24 | z << 16 | y << 8 | x;
  }
}
