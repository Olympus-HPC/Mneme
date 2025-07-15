#include <vector>
#include <iostream>
#include <cmath>
#include "reference.h"

HogbomGolden::HogbomGolden()
{
}

HogbomGolden::~HogbomGolden()
{
}

void HogbomGolden::deconvolve(const std::vector<float>& dirty,
    const size_t dirtyWidth,
    const std::vector<float>& psf,
    const size_t psfWidth,
    std::vector<float>& model,
    std::vector<float>& residual)
{
  residual = dirty;

  // Find the peak of the PSF
  float psfPeakVal = 0.0;
  size_t psfPeakPos = 0;
  findPeak(psf, psfPeakVal, psfPeakPos);
  std::cout << "Found peak of PSF: " << "Maximum = " << psfPeakVal
    << " at location " << idxToPos(psfPeakPos, psfWidth).x << ","
    << idxToPos(psfPeakPos, psfWidth).y << std::endl;

  for (unsigned int i = 0; i < niters; ++i) {
    // Find the peak in the residual image
    float absPeakVal = 0.0;
    size_t absPeakPos = 0;
    findPeak(residual, absPeakVal, absPeakPos);
    //printf("%d %f\n", absPeakPos, absPeakVal);

    // Check if threshold has been reached
    if (fabsf(absPeakVal) < threshold) {
      std::cout << "Reached stopping threshold" << std::endl;
      break;
    }

    // Add to model
    model[absPeakPos] += absPeakVal * gain;

    // Subtract the PSF from the residual image
    subtractPSF(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal, gain);
  }
}

void HogbomGolden::subtractPSF(const std::vector<float>& psf,
    const size_t psfWidth,
    std::vector<float>& residual,
    const size_t residualWidth,
    const size_t peakPos, const size_t psfPeakPos,
    const float absPeakVal,
    const float gain)
{
  const int rx = idxToPos(peakPos, residualWidth).x;
  const int ry = idxToPos(peakPos, residualWidth).y;

  const int px = idxToPos(psfPeakPos, psfWidth).x;
  const int py = idxToPos(psfPeakPos, psfWidth).y;

  const int diffx = rx - px;
  const int diffy = ry - px;

  const int startx = std::max(0, rx - px);
  const int starty = std::max(0, ry - py);

  const int stopx = std::min(residualWidth - 1, rx + (psfWidth - px - 1));
  const int stopy = std::min(residualWidth - 1, ry + (psfWidth - py - 1));

  for (int y = starty; y <= stopy; ++y) {
    for (int x = startx; x <= stopx; ++x) {
      residual[posToIdx(residualWidth, Position(x, y))] -= gain * absPeakVal
        * psf[posToIdx(psfWidth, Position(x - diffx, y - diffy))];
    }
  }
}

void HogbomGolden::findPeak(const std::vector<float>& image,
    float& maxVal, size_t& maxPos)
{
  maxVal = 0.0;
  maxPos = 0;
  const size_t size = image.size();
  for (size_t i = 0; i < size; ++i) {
    if (fabsf(image[i]) > fabsf(maxVal)) {
      maxVal = image[i];
      maxPos = i;
    }
  }
}

HogbomGolden::Position HogbomGolden::idxToPos(const int idx, const size_t width)
{
  const int y = idx / width;
  const int x = idx % width;
  return Position(x, y);
}

size_t HogbomGolden::posToIdx(const size_t width, const HogbomGolden::Position& pos)
{
  return (pos.y * width) + pos.x;
}
