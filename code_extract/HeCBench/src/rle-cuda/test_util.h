/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


#pragma once

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
    #include <sys/resource.h>
#endif

#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstddef>

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <limits>

#include "mersenne.h"

#include <cub/util_debug.cuh>
#include <cub/util_macro.cuh>

/**
 * Helper for casting character types to integers for cout printing
 */
template <typename T>
T CoutCast(T val) { return val; }

inline int CoutCast(char val) { return val; }

inline int CoutCast(unsigned char val) { return val; }

inline int CoutCast(signed char val) { return val; }

/******************************************************************************
 * Type conversion macros
 ******************************************************************************/

/**
 * Return a value of type `T` with the same bitwise representation of `in`.
 * Types `T` and `U` must be the same size.
 */
template <typename T, typename U>
T SafeBitCast(const U& in)
{
  static_assert(sizeof(T) == sizeof(U), "Types must be same size.");
  T out;
  memcpy(&out, &in, sizeof(T));
  return out;
}

/******************************************************************************
 * Assertion macros
 ******************************************************************************/

/**
 * Assert equals
 */
#define AssertEquals(a, b)                                                     \
  if ((a) != (b))                                                              \
  {                                                                            \
    std::cerr << "\n"                                                          \
              << __FILE__ << ": " << __LINE__                                  \
              << ": AssertEquals(" #a ", " #b ") failed.\n";                   \
    exit(1);                                                                   \
  }

#define AssertTrue(a)                                                          \
  if (!(a))                                                                    \
  {                                                                            \
    std::cerr << "\n"                                                          \
              << __FILE__ << ": " << __LINE__                                  \
              << ": AssertTrue(" #a ") failed.\n";                             \
    exit(1);                                                                   \
  }

/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
struct CommandLineArgs
{

    std::vector<std::string>    keys;
    std::vector<std::string>    values;
    std::vector<std::string>    args;

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv) :
        keys(10),
        values(10)
    {
        using namespace std;

        // Initialize mersenne generator
        unsigned int mersenne_init[4]=  {0x123, 0x234, 0x345, 0x456};
        mersenne::init_by_array(mersenne_init, 4);

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-'))
            {
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find('=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;

        for (std::size_t i = 0; i < keys.size(); ++i)
        {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }


    /**
     * Returns number of naked (non-flag and non-key-value) commandline parameters
     */
    template <typename T>
    int NumNakedArgs()
    {
        return args.size();
    }


    /**
     * Returns the commandline parameter for a given index (not including flags)
     */
    template <typename T>
    void GetCmdLineArgument(std::size_t index, T &val)
    {
        using namespace std;
        if (index < args.size()) {
            istringstream str_stream(args[index]);
            str_stream >> val;
        }
    }

    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;

        for (std::size_t i = 0; i < keys.size(); ++i)
        {
            if (keys[i] == string(arg_name))
            {
                istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }


    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        using namespace std;

        if (CheckCmdLineFlag(arg_name))
        {
            // Clear any default values
            vals.clear();

            // Recover from multi-value string
            for (std::size_t i = 0; i < keys.size(); ++i)
            {
                if (keys[i] == string(arg_name))
                {
                    string val_string(values[i]);
                    istringstream str_stream(val_string);
                    string::size_type old_pos = 0;
                    string::size_type new_pos = 0;

                    // Iterate comma-separated values
                    T val;
                    while ((new_pos = val_string.find(',', old_pos)) != string::npos)
                    {
                        if (new_pos != old_pos)
                        {
                            str_stream.width(new_pos - old_pos);
                            str_stream >> val;
                            vals.push_back(val);
                        }

                        // skip over comma
                        str_stream.ignore(1);
                        old_pos = new_pos + 1;
                    }

                    // Read last value
                    str_stream >> val;
                    vals.push_back(val);
                }
            }
        }
    }


    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return (int) keys.size();
    }
};

/******************************************************************************
 * Random bits generator
 ******************************************************************************/

int g_num_rand_samples = 0;


template <typename T>
bool IsNaN(T /* val */) { return false; }

template<>
__noinline__ bool IsNaN<float>(float val)
{
  return std::isnan(val);
}

template<>
__noinline__ bool IsNaN<double>(double val)
{
  return std::isnan(val);
}

/**
 * Generates random keys.
 *
 * We always take the second-order byte from rand() because the higher-order
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 *
 * We can decrease the entropy level of keys by adopting the technique
 * of Thearling and Smith in which keys are computed from the bitwise AND of
 * multiple random samples:
 *
 * entropy_reduction    | Effectively-unique bits per key
 * -----------------------------------------------------
 * -1                   | 0
 * 0                    | 32
 * 1                    | 25.95 (81%)
 * 2                    | 17.41 (54%)
 * 3                    | 10.78 (34%)
 * 4                    | 6.42 (20%)
 * ...                  | ...
 *
 */
template <typename K>
void RandomBits(
    K &key,
    int entropy_reduction = 0,
    int begin_bit = 0,
    int end_bit = sizeof(K) * 8)
{
    const int NUM_BYTES = sizeof(K);
    const int WORD_BYTES = sizeof(unsigned int);
    const int NUM_WORDS = (NUM_BYTES + WORD_BYTES - 1) / WORD_BYTES;

    unsigned int word_buff[NUM_WORDS];

    if (entropy_reduction == -1)
    {
        memset((void *) &key, 0, sizeof(key));
        return;
    }

    if (end_bit < 0)
        end_bit = sizeof(K) * 8;

    while (true)
    {
        // Generate random word_buff
        for (int j = 0; j < NUM_WORDS; j++)
        {
            int current_bit = j * WORD_BYTES * 8;

            unsigned int word = 0xffffffff;
            word &= 0xffffffff << CUB_MAX(0, begin_bit - current_bit);
            word &= 0xffffffff >> CUB_MAX(0, (current_bit + (WORD_BYTES * 8)) - end_bit);

            for (int i = 0; i <= entropy_reduction; i++)
            {
                // Grab some of the higher bits from rand (better entropy, supposedly)
                word &= mersenne::genrand_int32();
                g_num_rand_samples++;
            }

            word_buff[j] = word;
        }

        memcpy(&key, word_buff, sizeof(K));

        K copy = key;
        if (!IsNaN(copy))
            break;          // avoids NaNs when generating random floating point numbers
    }
}

/// Randomly select number between [0:max)
template <typename T>
T RandomValue(T max)
{
    unsigned int bits;
    unsigned int max_int = (unsigned int) -1;
    do {
        RandomBits(bits);
    } while (bits == max_int);

    return (T) ((double(bits) / double(max_int)) * double(max));
}


/******************************************************************************
 * Comparison and ostream operators
 ******************************************************************************/


/******************************************************************************
 * Helper routines for list comparison and display
 ******************************************************************************/

/**
 * Compares the equivalence of two arrays
 */
template <typename S, typename T, typename OffsetT>
int CompareResults(T* computed, S* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                << CoutCast(computed[i]) << " != "
                << CoutCast(reference[i]);
            return 1;
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(float* computed, float* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            float difference = std::abs(computed[i]-reference[i]);
            float fraction = difference / std::abs(reference[i]);

            if (fraction > 0.00015)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << "(computed) " << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Compares the equivalence of two arrays
 */
template <typename OffsetT>
int CompareResults(double* computed, double* reference, OffsetT len, bool verbose = true)
{
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            double difference = std::abs(computed[i]-reference[i]);
            double fraction = difference / std::abs(reference[i]);

            if (fraction > 0.00015)
            {
                if (verbose) std::cout << "INCORRECT: [" << i << "]: "
                    << CoutCast(computed[i]) << " != "
                    << CoutCast(reference[i]) << " (difference:" << difference << ", fraction: " << fraction << ")";
                return 1;
            }
        }
    }
    return 0;
}


/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename S, typename T>
int CompareDeviceResults(
    S *h_reference,
    T *d_data,
    std::size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    if (num_items == 0)
    {
        return 0;
    }

    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data)
    {
        printf("Reference:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nComputed:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_data) free(h_data);

    return retval;
}


/**
 * Verify the contents of a device array match those
 * of a device array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T *d_reference,
    T *d_data,
    std::size_t num_items,
    bool verbose = true,
    bool display_data = false)
{
    // Allocate array on host
    T *h_reference = (T*) malloc(num_items * sizeof(T));
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_reference, d_reference, sizeof(T) * num_items, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    // Display data
    if (display_data) {
        printf("Reference:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_reference[i]) << ", ";
        }
        printf("\n\nComputed:\n");
        for (std::size_t i = 0; i < num_items; i++)
        {
            std::cout << CoutCast(h_data[i]) << ", ";
        }
        printf("\n\n");
    }

    // Check
    int retval = CompareResults(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_reference) free(h_reference);
    if (h_data) free(h_data);

    return retval;
}

/**
 * Print the contents of a host array
 */
template <typename InputIteratorT>
void DisplayResults(
    InputIteratorT h_data,
    std::size_t num_items)
{
    // Display data
    for (std::size_t i = 0; i < num_items; i++)
    {
        std::cout << CoutCast(h_data[i]) << ", ";
    }
    printf("\n");
}


/**
 * Print the contents of a device array
 */
template <typename T>
void DisplayDeviceResults(
    T *d_data,
    std::size_t num_items)
{
    // Allocate array on host
    T *h_data = (T*) malloc(num_items * sizeof(T));

    // Copy data back
    cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost);

    DisplayResults(h_data, num_items);

    // Cleanup
    if (h_data) free(h_data);
}


/******************************************************************************
 * Segment descriptor generation
 ******************************************************************************/

struct HugeDataType
{
  static constexpr int ELEMENTS_PER_OBJECT = 128;

  __device__ __host__ HugeDataType()
  {
    for (int i = 0; i < ELEMENTS_PER_OBJECT; i++)
    {
      data[i] = 0;
    }
  }

  __device__ __host__ HugeDataType(const HugeDataType&rhs)
  {
    for (int i = 0; i < ELEMENTS_PER_OBJECT; i++)
    {
      data[i] = rhs.data[i];
    }
  }

  explicit __device__ __host__ HugeDataType(int val)
  {
    for (int i = 0; i < ELEMENTS_PER_OBJECT; i++)
    {
      data[i] = val;
    }
  }

  int data[ELEMENTS_PER_OBJECT];
};

inline __device__ __host__ bool operator==(const HugeDataType &lhs,
                                           const HugeDataType &rhs)
{
  for (int i = 0; i < HugeDataType::ELEMENTS_PER_OBJECT; i++)
  {
    if (lhs.data[i] != rhs.data[i])
    {
      return false;
    }
  }

  return true;
}

inline __device__ __host__ bool operator<(const HugeDataType &lhs,
                                          const HugeDataType &rhs)
{
  for (int i = 0; i < HugeDataType::ELEMENTS_PER_OBJECT; i++)
  {
    if (lhs.data[i] < rhs.data[i])
    {
      return true;
    }
  }

  return false;
}

template <typename DataType>
__device__ __host__ bool operator!=(const HugeDataType &lhs,
                                    const DataType &rhs)
{
  for (int i = 0; i < HugeDataType::ELEMENTS_PER_OBJECT; i++)
  {
    if (lhs.data[i] != rhs)
    {
      return true;
    }
  }

  return false;
}
