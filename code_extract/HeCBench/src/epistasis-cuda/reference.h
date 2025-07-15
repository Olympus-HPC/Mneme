unsigned pc (unsigned x)
{
  unsigned count;
  for (count=0; x; count++)
    x &= x - 1;
  return count;
}

float gamma(unsigned int n)
{
  if(n == 0)
    return 0.0f;
  float x = ((float)n + 0.5f) * logf((float) n) -
    ((float)n - 1.0f) * logf(expf(1.0f));
  return x;
}

void reference(
    const unsigned int* data_zeros,
    const unsigned int* data_ones,
    float* scores,
    const int num_snp,
    const int PP_zeros,
    const int PP_ones,
    const int mask_zeros,
    const int mask_ones)
{
  int p, k;

  for (int i = 0; i < num_snp; i++) {
    for (int j = 0; j < num_snp; j++) {

      if (j > i) {
        unsigned int ft[2 * 9];
        for(k = 0; k < 2 * 9; k++) ft[k] = 0;

        unsigned int t00, t01, t02, t10, t11, t12, t20, t21, t22;
        unsigned int di2, dj2;
        unsigned int* SNPi;
        unsigned int* SNPj;

        // Phenotype 0
        SNPi = (unsigned int*) &data_zeros[i * 2];
        SNPj = (unsigned int*) &data_zeros[j * 2];
        for (p = 0; p < 2 * PP_zeros * num_snp - 2 * num_snp; p += 2 * num_snp) {
          di2 = ~(SNPi[p] | SNPi[p + 1]);
          dj2 = ~(SNPj[p] | SNPj[p + 1]);

          t00 = SNPi[p] & SNPj[p];
          t01 = SNPi[p] & SNPj[p + 1];
          t02 = SNPi[p] & dj2;
          t10 = SNPi[p + 1] & SNPj[p];
          t11 = SNPi[p + 1] & SNPj[p + 1];
          t12 = SNPi[p + 1] & dj2;
          t20 = di2 & SNPj[p];
          t21 = di2 & SNPj[p + 1];
          t22 = di2 & dj2;

          ft[0] += pc(t00);
          ft[1] += pc(t01);
          ft[2] += pc(t02);
          ft[3] += pc(t10);
          ft[4] += pc(t11);
          ft[5] += pc(t12);
          ft[6] += pc(t20);
          ft[7] += pc(t21);
          ft[8] += pc(t22);
        }

        // remainder
        p = 2 * PP_zeros * num_snp - 2 * num_snp;
        di2 = ~(SNPi[p] | SNPi[p + 1]);
        dj2 = ~(SNPj[p] | SNPj[p + 1]);
        di2 = di2 & mask_zeros;
        dj2 = dj2 & mask_zeros;

        t00 = SNPi[p] & SNPj[p];
        t01 = SNPi[p] & SNPj[p + 1];
        t02 = SNPi[p] & dj2;
        t10 = SNPi[p + 1] & SNPj[p];
        t11 = SNPi[p + 1] & SNPj[p + 1];
        t12 = SNPi[p + 1] & dj2;
        t20 = di2 & SNPj[p];
        t21 = di2 & SNPj[p + 1];
        t22 = di2 & dj2;

        ft[0] += pc(t00);
        ft[1] += pc(t01);
        ft[2] += pc(t02);
        ft[3] += pc(t10);
        ft[4] += pc(t11);
        ft[5] += pc(t12);
        ft[6] += pc(t20);
        ft[7] += pc(t21);
        ft[8] += pc(t22);

        // Phenotype 1
        SNPi = (unsigned int*) &data_ones[i * 2];
        SNPj = (unsigned int*) &data_ones[j * 2];
        for(p = 0; p < 2 * PP_ones * num_snp - 2 * num_snp; p += 2 * num_snp)
        {
          di2 = ~(SNPi[p] | SNPi[p + 1]);
          dj2 = ~(SNPj[p] | SNPj[p + 1]);

          t00 = SNPi[p] & SNPj[p];
          t01 = SNPi[p] & SNPj[p + 1];
          t02 = SNPi[p] & dj2;
          t10 = SNPi[p + 1] & SNPj[p];
          t11 = SNPi[p + 1] & SNPj[p + 1];
          t12 = SNPi[p + 1] & dj2;
          t20 = di2 & SNPj[p];
          t21 = di2 & SNPj[p + 1];
          t22 = di2 & dj2;

          ft[9]  += pc(t00);
          ft[10] += pc(t01);
          ft[11] += pc(t02);
          ft[12] += pc(t10);
          ft[13] += pc(t11);
          ft[14] += pc(t12);
          ft[15] += pc(t20);
          ft[16] += pc(t21);
          ft[17] += pc(t22);
        }
        p = 2 * PP_ones * num_snp - 2 * num_snp;
        di2 = ~(SNPi[p] | SNPi[p + 1]);
        dj2 = ~(SNPj[p] | SNPj[p + 1]);
        di2 = di2 & mask_ones;
        dj2 = dj2 & mask_ones;

        t00 = SNPi[p] & SNPj[p];
        t01 = SNPi[p] & SNPj[p + 1];
        t02 = SNPi[p] & dj2;
        t10 = SNPi[p + 1] & SNPj[p];
        t11 = SNPi[p + 1] & SNPj[p + 1];
        t12 = SNPi[p + 1] & dj2;
        t20 = di2 & SNPj[p];
        t21 = di2 & SNPj[p + 1];
        t22 = di2 & dj2;

        ft[9]  += pc(t00);
        ft[10] += pc(t01);
        ft[11] += pc(t02);
        ft[12] += pc(t10);
        ft[13] += pc(t11);
        ft[14] += pc(t12);
        ft[15] += pc(t20);
        ft[16] += pc(t21);
        ft[17] += pc(t22);

        // compute score
        float score = 0.0f;
        for(k = 0; k < 9; k++)
          score += gamma(ft[k] + ft[9 + k] + 1) - gamma(ft[k]) - gamma(ft[9 + k]);
        score = fabsf(score);
        if(score == 0.0f)
          score = FLT_MAX;
        scores[i * num_snp + j] = score;
      }
    }
  }
}

int min_score(const float *scores, const int nrows, const int ncols) {
  // compute the minimum score on a host
  float score = scores[0];
  int solution = 0;
  for (int i = 1; i < nrows * ncols; i++) {
    if (score > scores[i]) {
      score = scores[i];
      solution = i;
    }
  }
  return solution;
}
