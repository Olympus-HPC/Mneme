#define UTILITIES_CPP

#ifndef GRAPH_CPP
#include "graph.cpp"
#endif

// function to return a random myInteger with probability according to a given
// (normalized!) weights
myInt rand_myInt_weighted(myInt n, Real *weights) {
  myInt i;
  Real r = gsl_ran_flat(rnd, 0.0, 1.0);
  for (i = 0; i < n; i++) {
    if (r < weights[i]) {
      break;
    };
    r -= weights[i];
  };
  return i;
}

// returns a random myInterger between 0 and n - 1
myInt rand_myInt(myInt n) {
  Real alpha = gsl_ran_flat(rnd, 0.0, 1.0);
  myInt value = (myInt)(n * alpha) / 1;
  return (value);
}

// function to return a random myInteger with probability according to a given
// (normalized!) weights
int rand_int_weighted(int n, Real *weights) {
  int i;
  Real r = gsl_ran_flat(rnd, 0.0, 1.0);
  for (i = 0; i < n; i++) {
    if (r < weights[i]) {
      break;
    };
    r -= weights[i];
  };
  return i;
}

// returns a random myInterger between 0 and n - 1
int rand_int(int n) {
  Real alpha = gsl_ran_flat(rnd, 0.0, 1.0);
  int value = (int)(n * alpha) / 1;
  return (value);
}

// This samples an object from a subset of 0 to n - 1
int sample_from(int *s, int n, int n_sub) {
  int i;
  int k = 0;
  int sample = rand_int(n_sub);
  for (i = 0; i < n; i++) {
    if (s[i]) {
      if (k == sample) {
        return (i);
      };
      k++;
    }
  }

  return (-1);
}

void shuffle(int *randomorder, int start, int end, int size) {
  int i, j, k;
  for (i = start; i < end; i++) {
    j = rand_int(size);
    k = randomorder[j];
    randomorder[j] = randomorder[i];
    randomorder[i] = k;
  }
}