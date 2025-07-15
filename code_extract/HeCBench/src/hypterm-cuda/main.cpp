#include <cassert>
#include <cstdio>
#include "utils.h"

extern "C" void reference (double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int);
extern "C" void offload (double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double, double, double, int, int, int, int);

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  double (*cons_1)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*cons_2)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*cons_3)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*cons_4)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*q_1)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*q_2)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*q_3)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*q_4)[D][D] = (double (*)[D][D]) getRandom3DArray<double>(D, D, D);
  double (*flux_0)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_1)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_2)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_3)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_4)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_gold_0)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_gold_1)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_gold_2)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_gold_3)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double (*flux_gold_4)[D][D] = (double (*)[D][D]) getZero3DArray<double>(D, D, D);
  double *dxinv = (double*) malloc (sizeof (double) * 3);
  dxinv[0] = 0.01f;
  dxinv[1] = 0.02f;
  dxinv[2] = 0.03f;

  reference ((double*)flux_gold_0, (double*)flux_gold_1, (double*)flux_gold_2, (double*)flux_gold_3, (double*)flux_gold_4,
             (double*)cons_1, (double*)cons_2, (double*)cons_3, (double*)cons_4,
             (double*)q_1, (double*)q_2, (double*)q_3, (double*)q_4, dxinv, D);

  offload ((double*)flux_0, (double*)flux_1, (double*)flux_2, (double*)flux_3, (double*)flux_4,
           (double*)cons_1, (double*)cons_2, (double*)cons_3, (double*)cons_4,
           (double*)q_1, (double*)q_2, (double*)q_3, (double*)q_4, dxinv[0], dxinv[1], dxinv[2], D, D, D, repeat);

  double error;
  printf("Check flux_0\n");
  error = checkError3D<double> (D, D, (double*)flux_0, (double*)flux_gold_0, 4, D-4, 4, D-4, 4, D-4);
  printf("RMS Error : %e\n", error);

  printf("Check flux_1\n");
  error = checkError3D<double> (D, D, (double*)flux_1, (double*)flux_gold_1, 4, D-4, 4, D-4, 4, D-4);
  printf("RMS Error : %e\n", error);

  printf("Check flux_2\n");
  error = checkError3D<double> (D, D, (double*)flux_2, (double*)flux_gold_2, 4, D-4, 4, D-4, 4, D-4);
  printf("RMS Error : %e\n", error);

  printf("Check flux_3\n");
  error = checkError3D<double> (D, D, (double*)flux_3, (double*)flux_gold_3, 4, D-4, 4, D-4, 4, D-4);
  printf("RMS Error : %e\n", error);

  printf("Check flux_4\n");
  error = checkError3D<double> (D, D, (double*)flux_4, (double*)flux_gold_4, 4, D-4, 4, D-4, 4, D-4);
  printf("RMS Error : %e\n", error);

  delete[] cons_1;
  delete[] cons_2;
  delete[] cons_3;
  delete[] cons_4;
  delete[] q_1;
  delete[] q_2;
  delete[] q_3;
  delete[] q_4;
  delete[] flux_0;
  delete[] flux_1;
  delete[] flux_2;
  delete[] flux_3;
  delete[] flux_4;
  delete[] flux_gold_0;
  delete[] flux_gold_1;
  delete[] flux_gold_2;
  delete[] flux_gold_3;
  delete[] flux_gold_4;
 
  return 0;
}
