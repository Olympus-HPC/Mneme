#include "utils.h"

size_t align_value(size_t valueToAlign, size_t alignMask){
  if (valueToAlign % alignMask != 0) {
    valueToAlign = valueToAlign + (alignMask - valueToAlign % alignMask);
  }

  return valueToAlign;
}

void generateRandomValue(char *data, size_t size){
  srand(123);
  for (size_t idx = 0; idx < size; ++idx) 
    *(data+idx) = rand() % 256;
}

double elapsed_time_in_ms(struct timeval startTime, struct timeval endTime){
  return ((double)(endTime.tv_sec - startTime.tv_sec)) * 1000.0 + 
         ((double)(endTime.tv_usec - startTime.tv_usec)) / 1000.0;
}
