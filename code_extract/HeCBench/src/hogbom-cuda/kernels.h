#ifndef HOGBOM_TEST_H
#define HOGBOM_TEST_H

#include <vector>
#include <cstddef>

extern unsigned int niters;
extern const float gain;
extern const float threshold;

class HogbomTest {
  public:
    HogbomTest();
    ~HogbomTest();

    void deconvolve(const std::vector<float>& dirty,
        const size_t dirtyWidth,
        const std::vector<float>& psf,
        const size_t psfWidth,
        std::vector<float>& model,
        std::vector<float>& residual);
  private:
};

#endif
