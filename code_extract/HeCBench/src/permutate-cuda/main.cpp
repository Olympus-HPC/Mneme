#include "header.h"

int main(int argc, char* argv[])
{
  bool iidchk = 0;
  uint32_t samplesize = 1;
  uint32_t len = NUMBER_OF_SAMPLES;
  uint32_t numparallel = PARALLELISM;
  uint32_t numblock = BLOCK;
  uint32_t numthread = THREAD;
  bool verbose = VERBOSE;
  uint8_t *data;

  const char *in_file_name = argv[1];

  /* set the parameter and read the data from the input file */
  int status = input_by_user(&samplesize, &len, &numparallel, &numblock, &numthread, &verbose, in_file_name);
  if (status == 1) {
    return 1;
  }

  data = (uint8_t*)calloc(len, sizeof(uint8_t));
  if (data == NULL) {
     printf("Fail to allocate a data array\n");
     return 1;
  }
  read_data_from_file(data, samplesize, len, in_file_name);

  /* permutation testing */
  iidchk = permutation_testing(data, samplesize, len, numparallel, numblock, numthread, verbose);

  /* print the result of the permutation testing. */
  if (iidchk)
    printf("==> Assume that the noise source outputs are IID! \n");
  else
    printf("==> Reject the IID assumption! \n");

  free(data);

  return 0;
}


