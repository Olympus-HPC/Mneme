template <typename scalar_t, typename accscalar_t>
void sampleMultinomialOnce_cpu(
    int* dest,
    int distributions,
    int categories,
    const scalar_t* sampled,
    const scalar_t* dist,
    int stride_dist,
    int stride_categories)
{
  accscalar_t accZero = static_cast<accscalar_t>(0);
  scalar_t zero = static_cast<scalar_t>(0);
  int foundPos = 0;
  for (int curDist = 0; curDist < distributions; curDist++) {
    accscalar_t sum = accZero;
    scalar_t val;
    for (int cat = 0; cat < categories; cat++) {
      val = dist[curDist * stride_dist + cat * stride_categories];
      sum += static_cast<accscalar_t>(val);
    }
    if (sum == accZero) {
      dest[curDist] = 0;
      continue;
    }

    scalar_t sample = sampled[curDist];
    accscalar_t prevBucket;
    accscalar_t curBucket = accZero;
    bool found = false;
    for (int cat = 0; cat < categories; cat++) {
      accscalar_t dist_val = static_cast<accscalar_t>(dist[curDist * stride_dist + cat * stride_categories]) / sum ;
      prevBucket = curBucket;
      curBucket = prevBucket + dist_val;
      bool inBucket = ((sample < curBucket) && (sample >= prevBucket) &&
                      (dist_val > zero));
      if (inBucket) {
        foundPos = cat;
        found = true;
        break;
      }
    }

    if (found) {
      dest[curDist] = foundPos;
    } else {
      for (int cat = categories - 1; cat >= 0; --cat) {
        if (dist[curDist * stride_dist + cat * stride_categories] > zero) {
          dest[curDist] = cat;
          break;
        }
      }
    }
  }
}
