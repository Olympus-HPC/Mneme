#pragma once

#include <cassert>
#include <cstdint>
#include "hashers.cuh"

namespace kiss
{

/*! \brief KISS random number generator for host and CUDA device
* \tparam T base type of RNG state (\c std::uint32_t or \c std::uint64_t)
*/
template<class T>
class Kiss
{
    static_assert(
        std::is_same<T, std::uint32_t>::value ||
        std::is_same<T, std::uint64_t>::value,
        "invalid base type");

public:
    /*! \brief constructor
     * \param[in] random seed for initial state
     */
    __host__ __device__ inline
    constexpr explicit Kiss(T seed) noexcept
    {
        using Hasher = hashers::MurmurHash<T>;

        w = !seed ? 4294967295 : seed;

        // scramble until the initial state looks good
        // #pragma unroll 8 // throws warning in host code
        for (std::uint8_t iter = 0; iter < 8; iter++)
        {
            x = Hasher::hash(w);
            y = Hasher::hash(x);
            z = Hasher::hash(y);
            w = Hasher::hash(z);
        }

        // FIXME add proper backoff strategy
        assert(y != 0 && z != 0 && w != 0);
    }

    /*! \brief generator function
     * \tparam Result data type to be generated
     * \return random number
     */
    template<class Result>
    __host__ __device__ inline
    constexpr Result next() noexcept;

private:
    T x; //< partial state x
    T y; //< partial state y
    T z; //< partial state z
    T w; //< partial state w

}; // class Kiss

/*! \brief \c std::uint32_t generator
* \return uniform random \c std::uint32_t
*/
template< >
template< >
__host__ __device__ inline
constexpr std::uint32_t Kiss<std::uint32_t>::next<std::uint32_t>() noexcept
{
    // lcg
    x = 69069 * x + 12345;

    // xorshift
    y ^= y << 13;
    y ^= y >> 17;
    y ^= y << 5;

    // carry and multiply
    std::uint64_t t = 698769069ULL * z + w;
    w = t >> 32;
    z = (std::uint32_t) t;

    // combine
    return x + y + z;
}

/*! \brief \c std::uint64_t generator
* \return uniform random \c std::uint64_t
*/
template< >
template< >
__host__ __device__ inline
constexpr std::uint64_t Kiss<std::uint64_t>::next<std::uint64_t>() noexcept
{
    // lcg
    x = 6906969069ULL * x + 1234567;

    // xorshift
    y ^= y << 13;
    y ^= y >> 17;
    y ^= y << 43;

    // carry and multiply
    std::uint64_t t = (z << 58) + w;
    w = (z >> 6);
    z += t;
    w += (z < t);

    // combine
    return x + y + z;
}

/*! \brief float generator
* \return uniform random float in [0, 1)
*/
template< >
template< >
__host__ __device__ inline
constexpr float Kiss<std::uint32_t>::next<float>() noexcept
{
    std::uint32_t p = (next<std::uint32_t>() >> 9) | 0x3F800000;

    //float q = reinterpret_cast<float&>(p);
    // TODO waiting for C++20 std::bit_cast
    union reinterpreter_t
    {
        std::uint32_t from;
        float to;

        __host__ __device__
        constexpr reinterpreter_t(std::uint32_t from_) noexcept : from(from_) {}

    } reinterpreter(p);

    float q = reinterpreter.to;
    return q-1.0f;
}

/*! \brief double generator
* \return uniform random double in [0, 1)
*/
template< >
template< >
__host__ __device__ inline
constexpr double Kiss<std::uint32_t>::next<double>() noexcept
{
    std::uint32_t a = next<std::uint32_t>() >> 6;
    std::uint32_t b = next<std::uint32_t>() >> 5;
    double q = (a * 134217728.0 + b) / 9007199254740992.0;

    return q;
}

} // namespace kiss
