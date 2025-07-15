#include <algorithm>

template <class T>
static T bound(T value, T lower, T upper) {
  return std::min(std::max(value, lower), upper);
}

template <class T, int BINS>
void reference(int*__restrict__ indices_, 
              int*__restrict__ count_, 
              const T*__restrict__ scores_,
              const float threshold,
              const int classwise_topK,
              const int batch_size,
              const size_t num_classes,
              const size_t num_priors)
{
  for (int b = 0; b < batch_size; b++)
    for (size_t c = 0; c < num_classes; c++) {
      int bins[BINS];
      auto indices = indices_ + (b * num_classes + c) * classwise_topK;
      auto count = count_ + b * num_classes + c;
      auto scores = scores_ + (b * num_classes + c) * num_priors;
      for (int i = 0; i < BINS; i++) bins[i] = 0;
      for (size_t i = 0; i < num_priors; i++) {
        const float confidence = scores[i];
        if (confidence > threshold)
        {
          float conf_scaled = (confidence - threshold ) / (1.f - threshold);
          int bin_index = conf_scaled * BINS;
          bin_index = bound<int>(bin_index, 0, BINS - 1) - 1; // shift left by one
          if (bin_index >= 0) bins[bin_index]++;
        }
      }
    
      for (int i = BINS-1; i >= 1; i--) {
          bins[i-1] += bins[i];
      }
    
      for (size_t i = 0; i < num_priors; i++) {
        const float confidence = scores[i];
        if (confidence > threshold)
        {
          float conf_scaled = (confidence - threshold ) / (1.f - threshold);
          int bin_index = conf_scaled * BINS;
          bin_index = bound<int>(bin_index, 0, BINS - 1);
          const int idx = bins[bin_index];
          bins[bin_index]++;
          if (idx < classwise_topK)
          {
            indices[idx] = i;
            count[0]++;
          }
        }
      }
   }
}
