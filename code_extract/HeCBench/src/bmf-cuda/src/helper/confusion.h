#ifndef CONFUSION_H
#define CONFUSION_H

#include <cstdlib>

struct confusion_matrix {
  size_t TP;
  size_t TN;
  size_t FP;
  size_t FN;

  confusion_matrix() : TP(0), TN(0), FP(0), FN(0) {};

  confusion_matrix(size_t tp, size_t tn, size_t fp, size_t fn) : TP(tp), TN(tn), FP(fp), FN(fn) {};

  float precision() {
    return 1.0f*TP / (TP + FP);
  }

  float sensitivity() {
    return 1.0f*TP / (TP + FN);
  }

  float f1score() {
    return 2.0f*TP / (2*TP + FP + FN);
  }

  float jaccard() {
    return 1.0f*TP / (TP + FP + FN);
  }

  size_t total_error() {
    return FP + FN;
  }

  size_t problem_size() {
    return TP + TN + FP + FN;
  }

  float rel_error() {
    return float(total_error()) / problem_size();
  }
};

#endif
