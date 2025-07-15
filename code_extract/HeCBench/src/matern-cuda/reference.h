#define nsources 50
#define SX 16
#define SY nsources

// nu = 2.5
void matern_kernel_reference (
  const int num_sources,
  const int num_targets,
  const float l,
  const float *sources,
  const float *targets,
  const float *weights,
        float *result)
{
  for (int t = 0; t < num_targets; t++) {
    float sum = 0.f;
    for (int s = 0; s < num_sources; s++) {
      float squared_diff = 0.f;
      for (int i = 0; i < 3; i++) {
        squared_diff += (sources[s*3+i] - targets[t*3+i]) *
                        (sources[s*3+i] - targets[t*3+i]);
      }
      float diff = sqrtf(squared_diff);
      sum += (1.f + sqrtf(5.f) * diff / l + 5.f * squared_diff / (3.f * l * l)) *  
             expf(-sqrtf(5.f) * diff  / l) * weights[s];
    }
    result[t] = sum;
  }
}

