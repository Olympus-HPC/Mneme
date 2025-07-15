float* attention_host(const float* key, const float* value, const float* query,
                      const int n, const int d) 
{
  // intermediate
  float* dot_product = (float*) malloc (n * sizeof(float));
  float* score = (float*) malloc (n * sizeof(float));
  // result
  float* output = (float*) malloc (d * sizeof(float));

  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < d; j++)
      sum += key[i * d + j] * query[j];
    dot_product[i] = sum;
  }

  float sum = 0;
  for (int i = 0; i < n; i++)
    sum += expf(dot_product[i]);

  for (int i = 0; i < n; i++)
    score[i] = expf(dot_product[i]) / sum;

  for (int j = 0; j < d; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++)
      sum += score[i] * value[i * d + j];
    output[j] = sum;
  }

  free(dot_product);
  free(score);
  return output;
}


