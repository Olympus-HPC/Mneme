#include "header.h"

void calculate_statistics(double *dmean, double *dmedian, uint8_t *data, uint32_t size, uint32_t len)
{
  uint32_t i;
  *dmean = 0;
  for (i = 0; i < len; i++)
    *dmean += data[i];
  *dmean /= (double)len;

  if (size == 1)
    *dmedian = 0.5;
  else {
    vector<uint8_t> Data(data, data + len);
    sort(Data.begin(), Data.end());
    uint32_t half_len = len / 2;
    if (len % 2 == 0)
      *dmedian = (Data[half_len] + Data[half_len - 1]) / 2.0;
    else
      *dmedian = Data[half_len];
    Data.clear();
  }
}

int run_tests(double *results, double dmean, double dmedian, uint8_t *data, uint32_t size, uint32_t len)
{
  if (size == 1) {
    excursion_test(&results[0], dmean, data, len);
    runs_based_on_median(&results[4], &results[5], dmedian, data, len);
    compression(&results[18], data, len, size);

    uint32_t blen = len / 8;
    if ((len % 8) != 0)  blen++;
    uint8_t *bdata = (uint8_t*)calloc(blen, sizeof(uint8_t));
    if (bdata == NULL) {
      printf("Failed to allocate the bdata array\n");
      return 1;
    }

    conversion1(bdata, data, len);
    directional_runs_and_number_of_inc_dec(&results[1], &results[2], &results[3], bdata, blen);
    periodicity_covariance_test(&results[8], &results[13], bdata, blen, 1);
    periodicity_covariance_test(&results[9], &results[14], bdata, blen, 2);
    periodicity_covariance_test(&results[10], &results[15], bdata, blen, 8);
    periodicity_covariance_test(&results[11], &results[16], bdata, blen, 16);
    periodicity_covariance_test(&results[12], &results[17], bdata, blen, 32);

    conversion2(bdata, data, len);
    collision_test_statistic(&results[6], &results[7], bdata, 8, blen);

    free(bdata);
  }
  else {
    excursion_test(&results[0], dmean, data, len);
    directional_runs_and_number_of_inc_dec(&results[1], &results[2], &results[3], data, len);
    runs_based_on_median(&results[4], &results[5], dmedian, data, len);
    collision_test_statistic(&results[6], &results[7], data, size, len);
    periodicity_covariance_test(&results[8], &results[13], data, len, 1);
    periodicity_covariance_test(&results[9], &results[14], data, len, 2);
    periodicity_covariance_test(&results[10], &results[15], data, len, 8);
    periodicity_covariance_test(&results[11], &results[16], data, len, 16);
    periodicity_covariance_test(&results[12], &results[17], data, len, 32);
    compression(&results[18], data, len, size);
  }

  return 0;
}

void excursion_test(double *out, const double dmean, const uint8_t *data, const uint32_t len)
{
  double max = 0, temp = 0, sum = 0;
  uint32_t i = 0;

  for (i = 0; i < len; i++) {
    sum += data[i];
    temp = fabs(sum - ((i + 1)*dmean));
    if (max < temp)
      max = temp;
  }
  *out = max;
}

void directional_runs_and_number_of_inc_dec(double *out_num, double *out_len, double *out_max, const uint8_t *data, const uint32_t len)
{
  uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0, pos = 0;
  bool flag1 = 0, flag2 = 0;
  uint32_t i = 0;

  if (data[0] <= data[1])
    flag1 = 1;

  for (i = 1; i < len - 1; i++) {
    pos += flag1;
    flag2 = 0;

    if (data[i] <= data[i + 1])
      flag2 = 1;
    if (flag1 == flag2)
      len_runs++;
    else {
      num_runs++;
      if (len_runs > max_len_runs)
        max_len_runs = len_runs;
      len_runs = 1;
    }
    flag1 = flag2;
  }
  pos += flag1;
  *out_num = (double)num_runs;
  *out_len = (double)max_len_runs;
  *out_max = (double)max(pos, len - pos);
}

void runs_based_on_median(double *out_num, double *out_len, const double dmedian, const uint8_t *data, const uint32_t len)
{
  uint32_t num_runs = 1, len_runs = 1, max_len_runs = 0;
  bool flag1 = 0, flag2 = 0;
  uint32_t i = 0;

  if (data[0] >= dmedian)
    flag1 = 1;

  for (i = 1; i < len; i++) {
    flag2 = 0;

    if (data[i] >= dmedian)
      flag2 = 1;
    if (flag1 == flag2)
      len_runs++;
    else {
      num_runs++;
      if (len_runs > max_len_runs)
        max_len_runs = len_runs;
      len_runs = 1;
    }
    flag1 = flag2;
  }

  *out_num = (double)num_runs;
  *out_len = (double)max_len_runs;
}

int collision_test_statistic(double *out_avg, double *out_max, const uint8_t *data, const uint32_t size, const uint32_t len)
{
  uint32_t i = 0, j = 0, k = 0;
  bool *dups = (bool*)calloc((1 << size), sizeof(bool));
  if (dups == NULL) {
    printf("Failed to allocate an array for the collision test\n");
    return 1;
  }
  uint32_t cnt = 0;
  uint32_t max = 0, collision = 0;
  double avg = 0;

  while (i + j < len) {
    for (k = 0; k < (1U << size); k++) dups[k] = false;

    while (i + j < len) {
      if (dups[data[i + j]]) {
        collision = j;
        avg += collision;
        if (collision > max)
          max = collision;
        cnt++;
        i += j;
        j = 0;
        break;
      }
      else {
        dups[data[i + j]] = true;
        ++j;
      }
    }
    ++i;
  }

  *out_avg = avg / (double)cnt;
  *out_max = (double)max;

  free(dups);
  return 0;
}

void periodicity_covariance_test(double *out_num, double *out_strength, const uint8_t *data, const uint32_t len, uint32_t lag)
{
  double t1 = 0, t2 = 0;
  uint32_t i = 0;
  for (i = 0; i < len - lag; i++) {
    if (data[i] == data[i + lag])
      t1++;
    t2 += (data[i] * data[i + lag]);
  }

  *out_num = t1;
  *out_strength = t2;
}

void compression(double *out, const uint8_t *data, const uint32_t len, const uint32_t size)
{
  uint32_t max_symbol = (1 << size) - 1;

  char *msg;
  unsigned int curlen = 0;
  char *curmsg;

  // Build string of bytes
  // Reserve the necessary size sample_size*(floor(log10(max_symbol))+2)
  // This is "worst case" and accounts for the space at the end of the number, as well.
  msg = new char[(size_t)(floor(log10(max_symbol)) + 2.0)*len];
  msg[0] = '\0';
  curmsg = msg;

  for (uint32_t i = 0; i < len; ++i) {
    int res;
    res = sprintf(curmsg, "%u ", data[i]);
    curlen += res;
    curmsg += res;
  }

  if (curlen > 0) {
    // Remove the extra ' ' at the end
    curmsg--;
    *curmsg = '\0';
    curlen--;
  }

  // Set up structures for compression
  unsigned int dest_len = ceil(1.01*curlen) + 600;
  char* dest = new char[dest_len];

  // Compress and capture the size of the compressed data
  int rc = BZ2_bzBuffToBuffCompress(dest, &dest_len, msg, curlen, 5, 0, 0);

  // Free memory
  delete[](dest);
  delete[](msg);

  // Return with proper return code
  if (rc == BZ_OK) {
    *out = (double)dest_len;
  }
  else {
    *out = (double)0;
  }
}

void conversion1(uint8_t *bdata, const uint8_t *data, const uint32_t len)
{
  uint32_t i = 0, j = 0;
  for (i = 0; i < (len / 8); i++)
    for (j = 0; j < 8; j++)
      bdata[i] += data[8 * i + j];

  if ((len % 8) != 0)
    for (j = 0; j < (len % 8); j++)
      bdata[i] += data[len - j - 1];
}

void conversion2(uint8_t *bdata, const uint8_t *data, const uint32_t len)
{
  uint32_t i = 0, j = 0;
  for (i = 0; i < (len / 8); i++) {
    bdata[i] = 0;
    for (j = 0; j < 8; j++)
      bdata[i] ^= (data[8 * i + j] & 0x1) << (7 - j);
  }

  if ((len % 8) != 0) {
    bdata[i] = 0;
    for (j = 0; j < (len % 8); j++)
      bdata[i] += data[len - j - 1] << (7 + j - (len % 8));
  }
}
