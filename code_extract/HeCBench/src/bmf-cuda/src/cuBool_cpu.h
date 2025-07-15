#ifndef cuBool_CPU_H
#define cuBool_CPU_H

#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <omp.h>

#include "helper/rngpu.hpp"
#include "helper/confusion.h"

#include "config.h"
#include "io_and_allocation.hpp"
#include "bit_vector_functions.h"

using std::vector;
using std::min;

template<typename factor_t = uint32_t>
class cuBool
{
  public:
    using factor_matrix_t = vector<factor_t>;
    using bit_vector_t = uint32_t;
    using bit_matrix_t = vector<bit_vector_t>;

    using index_t = uint32_t;
    using error_t = float;
    using cuBool_config = cuBool_config<index_t, error_t>;
  private:
    struct factor_handler {
      factor_matrix_t A_;
      factor_matrix_t B_;
      error_t distance_;
      uint8_t factorDim_ = 20;
      size_t lineSize_ = 1;
      bool initialized_ = false;
    };

  public:
    cuBool(const bit_matrix_t& C,
        const index_t height,
        const index_t width,
        const float density,
        const size_t numActiveExperriments = 1)
    {
      std::cout << "~~~ CPU cuBool ~~~" << std::endl; 

      height_ = height;
      width_ = width;

      density_ = density;
      inverse_density_ = 1 / density;

      if(std::is_same<factor_t, uint32_t>::value) {
        lineSize_padded_ = 1;
      }
      else if(std::is_same<factor_t, float>::value) {
        lineSize_padded_ = 32;
      }

      max_parallel_lines_ = omp_get_max_threads();

      bestFactors = {};
      resetBest();
      std::cout << "Matrix dimensions:\t" << height_ << "x" << width_ << std::endl;

      initializeMatrix(C);

      if(initialized_) {
        std::cout << "cuBool initialization complete." << std::endl;
      } else {
        exit(1);
      }
      std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
    }

    ~cuBool() = default;

    bool resetBest() {
      bestFactors.distance_ = std::numeric_limits<error_t>::max();
      bestFactors.initialized_ = false;
      return true;
    }

    // allocate memory for matrix, factors and distances, copy matrix to device
    bool initializeMatrix(const bit_matrix_t& C)
    {
      if( SDIV(height_,32) * width_ != C.size()) {
        std::cerr << "cuBool construction: Matrix dimension mismatch." << std::endl;
        return false;
      }

      C_ = C;

      return initialized_ = true;
    }

  public:
    // initialize factors with custom Initializer function
    template<class Initializer>
    bool initializeFactors(const uint8_t factorDim, Initializer&& initilize) {
      auto& handler = activeFactors;

      handler.factorDim_ = factorDim;

      if(std::is_same<factor_t, uint32_t>::value) {
        handler.lineSize_ = 1;
      }
      else if(std::is_same<factor_t, float>::value) {
        handler.lineSize_ = handler.factorDim_;
      }

      initilize(handler);

      handler.distance_ = -1;

      return handler.initialized_ = true;
    }

    // initialize factors as copy of vectors
    bool initializeFactors(const factor_matrix_t& A, const factor_matrix_t& B, const uint8_t factorDim) {
      return initializeFactors(factorDim, [&,this](factor_handler& handler){
        if( A.size() != height_ * handler.lineSize_ || B.size() != width_ * handler.lineSize_) {
          std::cerr << "cuBool initialization: Factor dimension mismatch." << std::endl;
          return false;
        }

        activeFactors.A_ = A;
        activeFactors.B_ = B;
      });
    }

    // initialize factors on device according to INITIALIZATIONMODE
    bool initializeFactors(const uint8_t factorDim, uint32_t seed) {
      return initializeFactors(factorDim, [&,this](factor_handler& handler){
        float threshold = getInitChance(density_, handler.factorDim_);
        initFactor(activeFactors.A_, height_, activeFactors.factorDim_, seed, threshold);
        seed += height_;
        initFactor(activeFactors.B_, width_, activeFactors.factorDim_, seed, threshold);
      });
    }

    // initialize first factor with random base vectors and optimize other factor
    bool initializeFactorsRandomBase(const uint8_t factorDim, uint32_t seed) {
      return initializeFactors(factorDim, [&,this](factor_handler& handler){
        float threshold = getInitChance(density_, handler.factorDim_);
        initFactor(activeFactors.A_, height_, activeFactors.factorDim_, seed, threshold);

        // activeFactors.A_ = factor_matrix_t(height_, 0);
        activeFactors.B_ = factor_matrix_t(width_, 0);
        // #pragma omp parallel
        for(int k=0; k<activeFactors.factorDim_; ++k) {
          // updateWholeColumn(activeFactors.A_, height_, activeFactors.factorDim_, k, density_, seed);
          optimizeWholeColumn<true>(activeFactors.B_, width_, activeFactors.A_, height_, C_, activeFactors.factorDim_, k);
        }
        activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        std::cout << "Start distance: "
                  << "\tabs_err: " << activeFactors.distance_
                  << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                  << std::endl;

        // #pragma omp parallel
        for(int k=0; k<activeFactors.factorDim_; ++k) {
          // updateWholeColumn(activeFactors.A_, height_, activeFactors.factorDim_, k, density_, seed);
          optimizeWholeColumn<true>(activeFactors.B_, width_, activeFactors.A_, height_, C_, activeFactors.factorDim_, k);
        }
        activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
        std::cout << "Start distance: "
                  << "\tabs_err: " << activeFactors.distance_
                  << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                  << std::endl;
      });
    }

    bool verifyDistance() {
      if(!initialized_ && activeFactors.initialized_) {
        std::cerr << "cuBool not initialized." << std::endl;
        return false;
      }

      my_error_t distance_proof;

      distance_proof = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);

      bool equal = fabs(activeFactors.distance_- distance_proof) < 1e-3; // std::numeric_limits<float>::epsilon();
      if(!equal) {
        std::cout << "----- !Distances differ! -----\n";
        std::cout << "Running distance:  " << activeFactors.distance_ << "\n";
        std::cout << "Real distance:     " << distance_proof << std::endl;
      } else {
        std::cout << "Distance verified" << std::endl;
      }
      return equal;
    } 

    void getFactors(factor_matrix_t& A, factor_matrix_t& B) const {
      if(!activeFactors.initialized_) {
        std::cerr << "cuBool not initialized." << std::endl;
        return;
      }

      A = activeFactors.A_;
      B = activeFactors.B_;
    }

    my_error_t getDistance() const {
      if(!activeFactors.initialized_) {
        std::cerr << "cuBool not initialized." << std::endl;
        return -1;
      }
      return activeFactors.distance_;
    }

    void getBestFactors(factor_matrix_t& A, factor_matrix_t& B) const {
      if(!bestFactors.initialized_) {
        std::cerr << "cuBool not initialized." << std::endl;
        return;
      }

      A = bestFactors.A_;
      B = bestFactors.B_;
    }

    my_error_t getBestDistance() const {
      if(!bestFactors.initialized_) {
        std::cerr << "cuBool not initialized." << std::endl;
        return -1;
      }
      return bestFactors.distance_;
    }

    void runMultiple(const size_t numExperiments, const cuBool_config& config) {
      finalDistances.resize(numExperiments);

      fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);

      for(size_t i=0; i<numExperiments; ++i) {
        auto config_i = config;
        uint32_t seed;
        seed = fast_kiss32(state);
        config_i.seed = fast_kiss32(state);
        std::cout << "Starting run " << i << " with seed " << config_i.seed << std::endl;
        initializeFactors(config_i.factorDim, seed);
        finalDistances[i] = run(config_i);
      }
    }

    float run(const cuBool_config& config) {
      if(!initialized_) {
        std::cerr << "cuBool not initialized." << std::endl;
        return -1;
      }

      if(!activeFactors.initialized_) {
        std::cerr << "cuBool active factors not initialized." << std::endl;
        return -1;
      }

      activeFactors.distance_ = computeDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_, config.weight);

      if(config.verbosity > 0) {
        std::cout << "\tStart distance"
                  << "\tabs_err: " << activeFactors.distance_
                  << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                  << '\n';
      }

      index_t linesAtOnce = config.linesAtOnce;
      if(config.loadBalance) {
        linesAtOnce = linesAtOnce / max_parallel_lines_ * max_parallel_lines_;
        if (!linesAtOnce) linesAtOnce = max_parallel_lines_;
      }

      if(config.verbosity > 1) {
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n";
        std::cout << "- - - - Starting " << config.maxIterations
                  << " CPU iterations, changing " << linesAtOnce
                  << " lines each time\n";
        std::cout << "- - - - Showing error every " << config.distanceShowEvery
                  << " steps\n";
        if(config.tempStart > 0) {
          std::cout << "- - - - Start temperature " << config.tempStart
                    << " multiplied by " << config.reduceFactor
                    << " every " << config.reduceStep
                    << " steps\n";

        }
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
      }

      fast_kiss_state32_t state = get_initial_fast_kiss_state32(config.seed);
      float temperature = config.tempStart;
      float weight = config.weight;
      size_t iteration = 0;
      size_t iteration_master = 0;
      size_t stuckIterations = 0;
      auto distancePrev = activeFactors.distance_;
      my_error_t distance_update_sum = 0;
      index_t lineToBeChanged;
      uint32_t cpuSeed;

      #pragma omp parallel firstprivate(iteration)
      while( activeFactors.distance_ > config.distanceThreshold
             && iteration++ < config.maxIterations
             && temperature > config.tempEnd
             && stuckIterations < config.stuckIterationsBeforeBreak)
      {
        // Change rows
        #pragma omp single
        {
          lineToBeChanged = (fast_kiss32(state) % height_) / WARPSPERBLOCK * WARPSPERBLOCK;
          cpuSeed = fast_kiss32(state) + iteration;
        }

        my_error_t distance_update = vectorMatrixMultCompareLineCPU<false>(
            activeFactors.A_, height_, activeFactors.B_, width_, C_, activeFactors.factorDim_,
            lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
            config.flipManyChance, config.flipManyDepth,
            config.weight);

        // Change cols
        #pragma omp single
        {
          lineToBeChanged = (fast_kiss32(state) % width_) / WARPSPERBLOCK * WARPSPERBLOCK;
          cpuSeed = fast_kiss32(state) + iteration;
        }

        distance_update += vectorMatrixMultCompareLineCPU<true>(
            activeFactors.B_, width_, activeFactors.A_, height_, C_, activeFactors.factorDim_,
            lineToBeChanged, min(linesAtOnce, height_), cpuSeed, temperature/10,
            config.flipManyChance, config.flipManyDepth,
            config.weight);

        #pragma omp atomic
        distance_update_sum += distance_update;
        #pragma omp barrier


        #pragma omp single
        {
          // int hamming;
          if(iteration % config.distanceShowEvery == 0) {
            activeFactors.distance_ = computeHammingDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_);
          }

          if(distance_update_sum == distancePrev)
            stuckIterations++;
          else
            stuckIterations = 0;
          distancePrev = distance_update_sum;

          if(config.verbosity > 0 && iteration % config.distanceShowEvery == 0) {
            std::std::cout << "Iteration: " << iteration
                           << "\tupdate: " << distance_update_sum / config.distanceShowEvery
                           << "\thamming: " << activeFactors.distance_
                           << "\ttemp: " << temperature;
            std::std::cout << std::std::endl;

            distance_update_sum = 0;
          }

          if(iteration % config.reduceStep == 0) {
            temperature *= config.reduceFactor;
            if(weight > 1)
              weight *= config.reduceFactor;
            if(weight < 1)
              weight = 1;
          }

          iteration_master = iteration;
        }
      }

      if(config.verbosity > 0) {
        std::cout << "\tBreak condition:\t";
        if (!(iteration_master < config.maxIterations))
          std::cout << "Reached iteration limit: " << config.maxIterations;
        if (!(activeFactors.distance_ > config.distanceThreshold))
          std::cout << "Distance below threshold: " << config.distanceThreshold;
        if (!(temperature > config.tempEnd))
          std::cout << "Temperature below threshold";
        if (!(stuckIterations < config.stuckIterationsBeforeBreak))
          std::cout << "Stuck for " << stuckIterations << " iterations";
        std::cout << " after " << iteration_master << " iterations.\n";
      }

      // use hamming distance for final judgement
      activeFactors.distance_ = computeDistanceCPU(activeFactors.A_, activeFactors.B_, C_, height_, width_, 1);

      if(config.verbosity > 0) {
        std::cout << "\tFinal distance"
                  << "\tabs_err: " << activeFactors.distance_
                  << "\trel_err: " << float(activeFactors.distance_) / height_ / width_
                  << std::endl;
      }

      if(activeFactors.distance_ < bestFactors.distance_) {
        if(config.verbosity > 0) {
          std::cout << "\tResult is better than previous best. Saving new best." << std::endl;
        }

        bestFactors = activeFactors;
      }

      return float(activeFactors.distance_) / height_ / width_;
    }  

    const vector<float>& getDistances() const {
      return finalDistances;
    }

  private:
    bool initialized_ = false;
    bit_matrix_t C_;
    vector<my_error_t> weights_rows_;
    vector<my_error_t> weights_cols_;
    float density_;
    int inverse_density_;
    index_t height_ = 0;
    index_t width_ = 0;
    size_t lineSize_padded_ = 1;
    int max_parallel_lines_;
    factor_handler bestFactors;
    factor_handler activeFactors;
    vector<float> finalDistances;
};

#endif
