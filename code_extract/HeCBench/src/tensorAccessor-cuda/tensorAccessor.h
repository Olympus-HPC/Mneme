#pragma once

#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define HOST         __host__
  #define DEVICE       __device__
  #define HOST_DEVICE  __host__ __device__
#else
  #define HOST
  #define DEVICE
  #define HOST_DEVICE
#endif

// The PtrTraits argument to the TensorAccessor/GenericPackedTensorAccessor
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.
template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

// The PtrTraits argument is only relevant to cuda to support `__restrict__` pointers.
template<typename T, size_t N, 
         template <typename U> class PtrTraits = DefaultPtrTraits,
         typename index_t = int64_t>
class TensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  HOST_DEVICE TensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_), sizes_(sizes_), strides_(strides_) {}

  HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  HOST_DEVICE PtrType data() {
    return data_;
  }
  HOST_DEVICE const PtrType data() const {
    return data_;
  }
protected:
  PtrType data_;
  const index_t* sizes_;
  const index_t* strides_;
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, N, PtrTraits, index_t>(data_,sizes_,strides_) {}

  HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }

  HOST_DEVICE const TensorAccessor<T, N-1, PtrTraits, index_t> operator[](index_t i) const {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T,1,PtrTraits,index_t> : public TensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_,sizes_,strides_) {}
  HOST_DEVICE T & operator[](index_t i) {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->data_[this->strides_[0]*i];
  }
  HOST_DEVICE const T & operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }
};


// GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on for CUDA `Tensor`s on the host
// and as
// In contrast to `TensorAccessor`s, they copy the strides and sizes on instantiation (on the host)
// in order to transfer them on the device when calling kernels.
// On the device, indexing of multidimensional tensors gives to `TensorAccessor`s.
// Use RestrictPtrTraits as PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and std::copy isn't available
// on the device, so those functions are host only.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_) {
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
  }

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : data_(data_) {
    for (int i = 0; i < N; i++) {
      this->sizes_[i] = sizes_[i];
      this->strides_[i] = strides_[i];
    }
  }

  HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  HOST_DEVICE PtrType data() {
    return data_;
  }
  HOST_DEVICE const PtrType data() const {
    return data_;
  }
protected:
  PtrType data_;
  index_t sizes_[N];
  index_t strides_[N];
  HOST void bounds_check_(index_t i) const {
    assert(0 <= i && i < index_t{N});
  }
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessor : public GenericPackedTensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  HOST GenericPackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  HOST GenericPackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }

  DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }

  /// Returns a PackedTensorAccessor of the same dimension after transposing the
  /// two dimensions given. Does not actually move elements; transposition is
  /// made by permuting the size/stride arrays. If the dimensions are not valid,
  /// asserts.
  HOST GenericPackedTensorAccessor<T, N, PtrTraits, index_t> transpose(
      index_t dim1,
      index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    GenericPackedTensorAccessor<T, N, PtrTraits, index_t> result(
        this->data_, this->sizes_, this->strides_);
    std::swap(result.strides_[dim1], result.strides_[dim2]);
    std::swap(result.sizes_[dim1], result.sizes_[dim2]);
    return result;
  }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class GenericPackedTensorAccessor<T,1,PtrTraits,index_t> : public GenericPackedTensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  HOST GenericPackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  HOST GenericPackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  DEVICE T& operator[](index_t i) {
    return this->data_[this->strides_[0] * i];
  }
  DEVICE const T& operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }
  // Same as in the general N-dimensional case, but note that in the
  // 1-dimensional case the returned PackedTensorAccessor will always be an
  // identical copy of the original
  HOST GenericPackedTensorAccessor<T, 1, PtrTraits, index_t> transpose(
      index_t dim1,
      index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    return GenericPackedTensorAccessor<T, 1, PtrTraits, index_t>(
        this->data_, this->sizes_, this->strides_);
  }
};

template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;
