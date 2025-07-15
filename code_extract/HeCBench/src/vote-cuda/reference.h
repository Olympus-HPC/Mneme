// Global types and parameters
#define VOTE_DATA_GROUP 4

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// Generate the test pattern for Tests 1 and 2
void genVoteTestPattern(unsigned int *VOTE_PATTERN, int size) {
  // For testing VOTE.Any (all of these threads will return 0)
  for (int i = 0; i < size / 4; i++) {
    VOTE_PATTERN[i] = 0x00000000;
  }

  // For testing VOTE.Any (1/2 these threads will return 1)
  for (int i = 2 * size / 8; i < 4 * size / 8; i++) {
    VOTE_PATTERN[i] = (i & 0x01) ? i : 0;
  }

  // For testing VOTE.all (1/2 of these threads will return 0)
  for (int i = 2 * size / 4; i < 3 * size / 4; i++) {
    VOTE_PATTERN[i] = (i & 0x01) ? 0 : i;
  }

  // For testing VOTE.all (all of these threads will return 1)
  for (int i = 3 * size / 4; i < 4 * size / 4; i++) {
    VOTE_PATTERN[i] = 0xffffffff;
  }
}

int checkErrors1(unsigned int *h_result, int start, int end, int warp_size,
                 const char *voteType) {
  int i, sum = 0;

  for (sum = 0, i = start; i < end; i++) {
    sum += h_result[i];
  }

  if (sum > 0) {
    printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);

    for (i = start; i < end; i++) {
      printf("%d", h_result[i]);
    }

    printf("%d values FAILED\n", sum);
  }

  return (sum > 0);
}

int checkErrors2(unsigned int *h_result, int start, int end, int warp_size,
                 const char *voteType) {
  int i, sum = 0;

  for (sum = 0, i = start; i < end; i++) {
    sum += h_result[i];
  }

  if (sum != warp_size) {
    printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);

    for (i = start; i < end; i++) {
      printf("%d", h_result[i]);
    }

    printf(" - FAILED\n");
  }

  return (sum != warp_size);
}

// Verification code for Kernel #1
int checkResultsVoteAnyKernel1(unsigned int *h_result, int size,
                               int warp_size) {
  int error_count = 0;

  error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4,
                              warp_size, "Vote.Any");
  error_count +=
      checkErrors2(h_result, VOTE_DATA_GROUP * warp_size / 4,
                   2 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");
  error_count +=
      checkErrors2(h_result, 2 * VOTE_DATA_GROUP * warp_size / 4,
                   3 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");
  error_count +=
      checkErrors2(h_result, 3 * VOTE_DATA_GROUP * warp_size / 4,
                   4 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.Any");

  printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

// Verification code for Kernel #2
int checkResultsVoteAllKernel2(unsigned int *h_result, int size,
                               int warp_size) {
  int error_count = 0;

  error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4,
                              warp_size, "Vote.All");
  error_count +=
      checkErrors1(h_result, VOTE_DATA_GROUP * warp_size / 4,
                   2 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");
  error_count +=
      checkErrors1(h_result, 2 * VOTE_DATA_GROUP * warp_size / 4,
                   3 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");
  error_count +=
      checkErrors2(h_result, 3 * VOTE_DATA_GROUP * warp_size / 4,
                   4 * VOTE_DATA_GROUP * warp_size / 4, warp_size, "Vote.All");

  printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

// Verification code for Kernel #3
int checkResultsVoteAnyKernel3(bool *hinfo, int size) {
  int i, error_count = 0;

  for (i = 0; i < size * 3; i++) {
    switch (i % 3) {
      case 0:

        // First warp should be all zeros.
        if (hinfo[i] != (i >= size * 1)) {
          error_count++;
        }

        break;

      case 1:

        // First warp and half of second should be all zeros.
        if (hinfo[i] != (i >= size * 3 / 2)) {
          error_count++;
        }

        break;

      case 2:

        // First two warps should be all zeros.
        if (hinfo[i] != (i >= size * 2)) {
          error_count++;
        }

        break;
    }
  }

  printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}
