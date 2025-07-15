#include "util.h"
#include "Kernel128_one.h"
#include "Kernel128_winograd.h"
#include "Kernel256_one.h"
#include "Kernel256_winograd.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage %s <mode> <repeat more than twice>\n", argv[0]);
    return 1;
  }
  const int mode = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  double time_total = 0, ktime_total = 0;
  double time, ktime;

  for (int i = 0; i < repeat; i++) {
    switch (mode) {
      case 0:
        kernel_128(time, ktime);
        break;
      case 1:
        kernel_256(time, ktime);
        break;
      case 2:
        kernel_128_1_in(time, ktime);
        break;
      case 3:
        kernel_128_1_out(time, ktime);
        break;
      case 4:
        kernel_256_1_in(time, ktime);
        break;
      case 5:
        kernel_256_1_out(time, ktime);
        break;
    }
    if (i > 1) {
       time_total += time;
       ktime_total += ktime;
    }
  }
  printf("Case %d: Average device offload time: [%lf us]\n", mode, time_total * 1e-3 / (repeat - 2));
  printf("        Average kernel time: [%lf us]\n", ktime_total * 1e-3 / (repeat - 2));
  
  return 0;
}
