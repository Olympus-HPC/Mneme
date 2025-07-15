
#include <iostream>
#include <fstream>
#include <sys/time.h>

#include "timer/timer.hpp"
#include "divergence.hpp"

template <int np, typename real>
void readVelocity(real v[np][np][2], std::istream *input) {
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < np; j++) {
      for(int k = 0; k < np; k++) {
        (*input) >> v[k][j][i];
      }
    }
  }
}

template <int np, typename real>
void readElement(element<np, real> &elem,
                 std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> elem.metdet[i][j];
      elem.rmetdet[i][j] = 1 / elem.metdet[i][j];
    }
  }
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
      for(int k = 0; k < np; k++) {
        for(int l = 0; l < np; l++) {
          (*input) >> elem.Dinv[l][k][i][j];
        }
      }
    }
  }
}

template <int np, typename real>
void readDerivative(derivative<np, real> &deriv,
                    std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> deriv.Dvv[j][i];
    }
  }
}

template <int np, typename real>
void readDivergence(real divergence[np][np],
                    std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> divergence[i][j];
    }
  }
}

constexpr const int DIMS = 2;

template <int np, typename real>
void compareDivergences(const real v[np][np][DIMS],
                        const element<np, real> &elem,
                        const derivative<np, real> &deriv,
                        const real divergence_e[np][np],
                        const int numtests) {
  Timer::Timer time_c;
  /* Initial run to prevent cache timing from affecting us
   */
  std::cout << "Divergence on the CPU\n";
  real divergence_c[np][np];
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, real>(v, deriv, elem, divergence_c);
  }

  time_c.startTimer();
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, real>(v, deriv, elem, divergence_c);
  }
  time_c.stopTimer();

  std::cout << "Divergence on the GPU\n";
  Timer::Timer time_f;
  real divergence_f[np][np];
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(v, deriv, elem, divergence_f);
  }
  time_f.startTimer();
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(v, deriv, elem, divergence_f);
  }
  time_f.stopTimer();
  std::cout << "Divergence Errors\n";
  std::cout << "CPU             GPU\n";
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      std::cout << divergence_c[i][j] - divergence_e[i][j]
                << "    "
                << divergence_f[i][j] - divergence_e[i][j]
                << "\n";
    }
    std::cout << "\n";
  }

  std::cout << "CPU Time:\n" << time_c
            << "\n\nGPU Time:\n" << time_f << "\n";
}

int main(int argc, char **argv) {
  constexpr const int NP = 4;
  real v[NP][NP][DIMS];
  element<NP, real> elem;
  derivative<NP, real> deriv;
  real divergence_e[NP][NP];
  {
    std::istream *input;
    if(argc > 1) {
      input = new std::ifstream(argv[1]);
    } else {
      input = &std::cin;
    }
    readVelocity(v, input);
    readElement(elem, input);
    readDerivative(deriv, input);
    readDivergence(divergence_e, input);
    if(argc > 1) {
      delete input;
    }
  }

  constexpr const int defNumTests = 1e5;
  const int numtests = (argc > 2) ? std::stoi(argv[2]) : defNumTests;
  compareDivergences(v, elem, deriv, divergence_e, numtests);
  return 0;
}
