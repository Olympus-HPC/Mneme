#pragma once

#define LOG_WARP_THREADS(unused) (5)
#define WARP_THREADS(unused) (1 << LOG_WARP_THREADS(0))
#define PTX_WARP_THREADS        WARP_THREADS(0)
#define PTX_LOG_WARP_THREADS    LOG_WARP_THREADS(0)

struct NullType
{
    //using value_type = NullType;

    template <typename T>
    __host__ __device__ __forceinline__ NullType& operator =(const T&) { return *this; }

    __host__ __device__ __forceinline__ bool operator ==(const NullType&) { return true; }

    __host__ __device__ __forceinline__ bool operator !=(const NullType&) { return false; }
};


/// Structure alignment
template <typename T>
struct AlignBytes
{
    struct Pad
    {
        T       val;
        char    byte;
    };

    enum
    {
        /// The "true CUDA" alignment of T in bytes
        ALIGN_BYTES = sizeof(Pad) - sizeof(T)
    };

    /// The "truly aligned" type
    typedef T Type;
};

#define __ALIGN_BYTES(t, b)         \
    template <> struct AlignBytes<t>    \
    { enum { ALIGN_BYTES = b }; typedef __align__(b) t Type; };

__ALIGN_BYTES(short4, 8)
__ALIGN_BYTES(ushort4, 8)
__ALIGN_BYTES(int2, 8)
__ALIGN_BYTES(uint2, 8)
__ALIGN_BYTES(long long, 8)
__ALIGN_BYTES(unsigned long long, 8)
__ALIGN_BYTES(float2, 8)
__ALIGN_BYTES(double, 8)
__ALIGN_BYTES(long2, 16)
__ALIGN_BYTES(ulong2, 16)
__ALIGN_BYTES(int4, 16)
__ALIGN_BYTES(uint4, 16)
__ALIGN_BYTES(float4, 16)
__ALIGN_BYTES(long4, 16)
__ALIGN_BYTES(ulong4, 16)
__ALIGN_BYTES(longlong2, 16)
__ALIGN_BYTES(ulonglong2, 16)
__ALIGN_BYTES(double2, 16)
__ALIGN_BYTES(longlong4, 16)
__ALIGN_BYTES(ulonglong4, 16)
__ALIGN_BYTES(double4, 16)

template <typename T> struct AlignBytes<volatile T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const volatile T> : AlignBytes<T> {};

template <bool Test, class T1, class T2>
using conditional_t = typename std::conditional<Test, T1, T2>::type;

/// Unit-words of data movement
template <typename T>
struct UnitWord
{
    enum {
        ALIGN_BYTES = AlignBytes<T>::ALIGN_BYTES
    };

    template <typename Unit>
    struct IsMultiple
    {
        enum {
            UNIT_ALIGN_BYTES    = AlignBytes<Unit>::ALIGN_BYTES,
            IS_MULTIPLE         = (sizeof(T) % sizeof(Unit) == 0) && (int(ALIGN_BYTES) % int(UNIT_ALIGN_BYTES) == 0)
        };
    };

    /// Biggest shuffle word that T is a whole multiple of and is not larger than
    /// the alignment of T
    using ShuffleWord = conditional_t<
      IsMultiple<int>::IS_MULTIPLE,
      unsigned int,
      conditional_t<IsMultiple<short>::IS_MULTIPLE,
                                 unsigned short,
                                 unsigned char>>;

    /// Biggest volatile word that T is a whole multiple of and is not larger than
    /// the alignment of T
    using VolatileWord =
      conditional_t<IsMultiple<long long>::IS_MULTIPLE,
                                 unsigned long long,
                                 ShuffleWord>;

    /// Biggest memory-access word that T is a whole multiple of and is not larger
    /// than the alignment of T
    using DeviceWord =
      conditional_t<IsMultiple<longlong2>::IS_MULTIPLE,
                                 ulonglong2,
                                 VolatileWord>;
};

template <typename T>
struct Uninitialized
{
    /// Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename UnitWord<T>::DeviceWord DeviceWord;

    static constexpr std::size_t DATA_SIZE = sizeof(T);
    static constexpr std::size_t WORD_SIZE = sizeof(DeviceWord);
    static constexpr std::size_t WORDS = DATA_SIZE / WORD_SIZE;

    /// Backing storage
    DeviceWord storage[WORDS];

    /// Alias
    __host__ __device__ __forceinline__ T& Alias()
    {
        return reinterpret_cast<T&>(*this);
    }
};

/**
 * \brief Returns the row-major linear thread identifier for a multidimensional thread block
 */
__device__ __forceinline__ int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y)) +
            ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x)) +
            threadIdx.x;
}
