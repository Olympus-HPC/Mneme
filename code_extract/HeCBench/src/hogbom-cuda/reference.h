#ifndef HOGBOM_GOLDEN_H
#define HOGBOM_GOLDEN_H

#include <vector>
#include <cstddef>

extern unsigned int niters;
extern const float gain;
extern const float threshold;

class HogbomGolden {
  public:
    HogbomGolden();
    ~HogbomGolden();

    void deconvolve(const std::vector<float>& dirty,
        const size_t dirtyWidth,
        const std::vector<float>& psf,
        const size_t psfWidth,
        std::vector<float>& model,
        std::vector<float>& residual);

  private:

    struct Position {
      Position(int _x, int _y) : x(_x), y(_y) { };
      int x;
      int y;
    };

    void findPeak(const std::vector<float>& image,
        float& maxVal, size_t& maxPos);

    void subtractPSF(const std::vector<float>& psf,
        const size_t psfWidth,
        std::vector<float>& residual,
        const size_t residualWidth,
        const size_t peakPos, const size_t psfPeakPos,
        const float absPeakVal, const float gain);

    Position idxToPos(const int idx, const size_t width);

    size_t posToIdx(const size_t width, const Position& pos);
};

#endif
