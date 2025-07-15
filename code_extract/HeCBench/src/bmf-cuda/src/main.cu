#include <vector>
#include <iostream>
#include <string>

#include "helper/clipp.h"
#include "io_and_allocation.hpp"
#include "bit_vector_functions.h"

#ifdef USE_GPU
#include "cuBool_gpu.cuh"
#else
#include "cuBool_cpu.h"
#endif

using std::string;
using std::vector;
using clipp::value;
using clipp::option;

using my_bit_vector_t = uint32_t; // only tested uint32_t
using my_cuBool = cuBool<my_bit_vector_t>;

int main(int argc, char **argv) {
  string filename;
  size_t numRuns = 1;
  my_cuBool::cuBool_config config;

  auto cli = (
    value("dataset file", filename),
    (option("-r") & value("runs", numRuns)) % "number of runs",
    (option("-v") & value("verbosity", config.verbosity)) % "verbosity",
    (option("-d") & value("dim", config.factorDim)) % "latent dimension",
    (option("-l") & value("lines", config.linesAtOnce)) % "number of lines to update per iteration",
    (option("-i") & value("iter", config.maxIterations)) % "maximum number of iterations",
    (option("-e") & value("err", config.distanceThreshold)) % "error threshold",
    (option("--show") & value("s", config.distanceShowEvery)) % "show distance every <s> iterations",
    (option("--ts") & value("start temp", config.tempStart)) % "start temperature",
    (option("--te") & value("end temp", config.tempEnd)) % "end temperature",
    (option("-w", "--weight") & value("weight", config.weight)) % "weight in error measure",
    (option("--factor") & value("factor", config.reduceFactor)) % "temperature/weight reduction factor",
    (option("--move") & value("move", config.reduceStep)) % "reduce temperature/weight every <move> iterations",
    (option("--seed") & value("seed", config.seed)) % "seed for pseudo random numbers",
    (option("--fc") & value("flip chance", config.flipManyChance)) % "chance to flip multiple bits",
    (option("--fd") & value("flip depth", config.flipManyDepth)) % "flip chance for each bit in multi flip (negative power of two)",
    (option("--stuck") & value("s", config.stuckIterationsBeforeBreak)) % "stop if stuck for <s> iterations"
  );

  auto parseResult = clipp::parse(argc, argv, cli);
  if(!parseResult) {
    auto fmt = clipp::doc_formatting{}.doc_column(30);
    std::cout << clipp::make_man_page(cli, argv[0], fmt);
    return 1;
  }

  std::cout << "verbosity " << config.verbosity << "\n"
            << "factorDim " << int(config.factorDim) << "\n"
            << "maxIterations " << config.maxIterations << "\n"
            << "linesAtOnce " << config.linesAtOnce << "\n"
            << "distanceThreshold " << config.distanceThreshold << "\n"
            << "distanceShowEvery " << config.distanceShowEvery << "\n"
            << "stuckIterationsBeforeBreak " << config.stuckIterationsBeforeBreak << "\n"
            << "tempStart " << config.tempStart << "\n"
            << "tempEnd " << config.tempEnd << "\n"
            << "reduceFactor " << config.reduceFactor << "\n"
            << "reduceStep " << config.reduceStep << "\n"
            << "seed " << config.seed << "\n"
            << "loadBalance " << config.loadBalance << "\n"
            << "flipManyChance " << config.flipManyChance << "\n"
            << "flipManyDepth " << config.flipManyDepth << "\n"
            << "weight " << config.weight << "\n";

  int height, width;
  float density;

  vector<my_bit_vector_t> A0_vec, B0_vec, C0_vec;

  if (filename.compare("test") == 0) {
    height = 5000;
    width = 5000;
    generate_random_matrix(height, width, config.factorDim, 4, A0_vec, B0_vec, C0_vec, density);
  } else {
    readInputFileData(filename, C0_vec, height, width, density);
  }

  vector<my_bit_vector_t> A_vec, B_vec;

  size_t numSlots = std::min(size_t(2), numRuns);
  auto cuBool = my_cuBool(C0_vec, height, width, density, numSlots);

  TIMERSTART(GPUKERNELLOOP)
  cuBool.runMultiple(numRuns, config);
  TIMERSTOP(GPUKERNELLOOP)

  cuBool.getBestFactors(A_vec, B_vec);

  const auto& distances = cuBool.getDistances();
  writeDistancesToFile(filename, distances);

  writeFactorsToFiles(filename + "_best", A_vec, B_vec, config.factorDim);

  auto confusion = computeErrorsCPU(A_vec, B_vec, C0_vec, height, width);

  std::cout << "true_positives: \t" << confusion.TP << '\t';
  std::cout << "true_negatives: \t" << confusion.TN << '\n';
  std::cout << "false_positives:\t" << confusion.FP << '\t';
  std::cout << "false_negatives:\t" << confusion.FN << '\n';
  std::cout << "total error:\t" << confusion.total_error() << '\t';
  std::cout << "rel error:\t" << confusion.rel_error() << '\n';
  std::cout << "precision:\t" << confusion.precision()*100 << " %\n";
  std::cout << "recall:   \t" << confusion.sensitivity()*100 << " %\n";
  std::cout << "F1 score: \t" << confusion.f1score() << std::endl;

  int count = nonzeroDimension(A_vec);
  std::cout << "A uses " << count << " of " << int(config.factorDim) << " columns" << std::endl;
  count = nonzeroDimension(B_vec);
  std::cout << "B uses " << count << " of " << int(config.factorDim) << " columns" << std::endl;

  return 0;
}

