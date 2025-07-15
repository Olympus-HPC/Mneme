#include "utils.h"
/**
 * @brief Get the maximum number of scratch space items that will be needed to
 * to perform reduction on the input.
 *
 * @param num The number of items in the input.
 *
 * @return The number of elements required.
 */
size_t getReduceScratchSpaceSize(size_t const num)
{
  // in the first round, each block will write one value, and then the next
  // round will launch and write one value per block. After that the spaces
  // will be re-used
  size_t const base
      = std::min(BLOCK_WIDTH, static_cast<int>(roundUpDiv(num, BLOCK_SIZE)))
        * sizeof(uint64_t);

  return base;
}

size_t requiredWorkspaceSize(size_t const num, const nvcompType_t type)
{
  // we need a space for min values, and a space for maximum values
  size_t const bytes
      = sizeOfnvcompType(type) * getReduceScratchSpaceSize(num) * 2;

  return bytes;
}

