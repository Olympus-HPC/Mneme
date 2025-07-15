#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/find.h>
#include "reference.h"

thrust::host_vector<char> mtf(const std::vector<char> &word)
{
  thrust::host_vector<char> h_list(256);
  thrust::device_vector<char> d_list(256);

  // store the mtf result since input word is read-only
  thrust::host_vector<char> h_word(word.size());

  // copy word from host to device
  thrust::device_vector<char> d_word(word);

  thrust::device_vector<char>::iterator iter;

  size_t counter;

  for (counter = 0; counter < word.size(); counter++)
  {
    // copy list from host to device
    thrust::copy(h_list.begin(), h_list.end(), d_list.begin());

    // find the location of the symbol in the list
    auto w = word[counter];
    iter = thrust::find(d_list.begin(), d_list.end(), w);

    // update the list when the first symbols are not the same
    if (h_list[0] != w)
    {
      // shift the sublist [begin, iter) right by one
      thrust::copy(d_list.begin(), iter, h_list.begin() + 1);
      h_list[0] = w;
    }
  }

  for (counter = 0; counter < h_list.size(); counter++)
  {
    iter = thrust::find(d_word.begin(), d_word.end(), h_list[counter]);
    while (iter != d_word.end())
    {
      // replace word symbol with its index (counter)
      *iter = counter;
      iter = thrust::find(iter + 1, d_word.end(), h_list[counter]);
    }
  }

  // copy from device to host
  h_word = d_word;

  return h_word;
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Usage: %s <string length> <repeat>\n", argv[0]);
    exit(1);
  }

  const size_t len = atol(argv[1]);
  const int repeat = atoi(argv[2]);
  const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::vector<char> word(len);

  srand(123);
  for (size_t i = 0; i < len; i++) word[i] = a[rand() % 52];

  auto d_result = mtf(word);
  auto h_result = reference(word);
  bool ok = d_result == h_result;
  if (ok) {
    printf("PASS\n");
  }
  else {
    if (len < 16) {
      printf("host: ");
      for (size_t i = 0; i < len; i++) 
        printf("%d ", h_result[i]);
      printf("\ndevice: ");
      for (size_t i = 0; i < len; i++) 
        printf("%d ", d_result[i]);
      printf("\n");
    }
    printf("FAIL\n");
    return 1;
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) mtf(word);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-9f) / repeat << " (s)\n";
  return 0;
}
