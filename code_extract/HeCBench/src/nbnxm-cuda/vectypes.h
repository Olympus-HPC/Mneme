#ifndef GMX_MATH_VECTYPES_H
#define GMX_MATH_VECTYPES_H

#include <cassert>
#include <cmath>
#include <algorithm>
#include <type_traits>

#define XX 0 /* Defines for indexing in */
#define YY 1 /* vectors                 */
#define ZZ 2
#define DIM 3 /* Dimension of vectors    */

typedef float real;

typedef real rvec[DIM];

typedef double dvec[DIM];

typedef real matrix[DIM][DIM];

typedef real tensor[DIM][DIM];

typedef int ivec[DIM];

namespace gmx
{

/*! \brief
 * C++ class for 3D vectors.
 *
 * \tparam ValueType  Type
 *
 * This class provides a C++ version of rvec/dvec/ivec that can be put into STL
 * containers etc.  It is more or less a drop-in replacement for `rvec` and
 * friends: it can be used in most contexts that accept the equivalent C type.
 * However, there is one case where explicit conversion is necessary:
 *  - An array of these objects needs to be converted with as_vec_array() (or
 *    convenience methods like as_rvec_array()).
 *
 * For the array conversion to work, the compiler should not add any extra
 * alignment/padding in the layout of this class;  that this actually works as
 * intended is tested in the unit tests.
 *
 * \inpublicapi
 */
template<typename ValueType>
class BasicVector
{
public:
    //! Underlying raw C array type (rvec/dvec/ivec).
    using RawArray = ValueType[DIM];

    // The code here assumes ValueType has been deduced as a data type like int
    // and not a pointer like int*. If there is a use case for a 3-element array
    // of pointers, the implementation will be different enough that the whole
    // template class should have a separate partial specialization. We try to avoid
    // accidental matching to pointers, but this assertion is a no-cost extra check.
    //
    // TODO: Use std::is_pointer_v when CUDA 11 is a requirement.
    static_assert(!std::is_pointer<std::remove_cv_t<ValueType>>::value,
                  "BasicVector value type must not be a pointer.");

    //! Constructs default (uninitialized) vector.
    BasicVector() {}
    //! Constructs a vector from given values.
    __host__ __device__
    BasicVector(ValueType x, ValueType y, ValueType z) : x_{ x, y, z } {}
    /*! \brief
     * Constructs a vector from given values.
     *
     * This constructor is not explicit to support implicit conversions
     * that allow, e.g., calling `std::vector<RVec>:``:push_back()` directly
     * with an `rvec` parameter.
     */
    BasicVector(const RawArray x) : x_{ x[XX], x[YY], x[ZZ] } {}
    //! Default copy constructor.
    BasicVector(const BasicVector& src) = default;
    //! Default copy assignment operator.
    BasicVector& operator=(const BasicVector& v) = default;
    //! Default move constructor.
    BasicVector(BasicVector&& src) noexcept = default;
    //! Default move assignment operator.
    BasicVector& operator=(BasicVector&& v) noexcept = default;
    //! Indexing operator to make the class work as the raw array.
    __host__ __device__
    ValueType& operator[](int i) { return x_[i]; }
    //! Indexing operator to make the class work as the raw array.
    __host__ __device__
    ValueType operator[](int i) const { return x_[i]; }
    //! Return whether all elements compare equal
    bool operator==(const BasicVector<ValueType>& right)
    {
        return x_[0] == right[0] && x_[1] == right[1] && x_[2] == right[2];
    }
    //! Return whether any elements compare unequal
    bool operator!=(const BasicVector<ValueType>& right)
    {
        return x_[0] != right[0] || x_[1] != right[1] || x_[2] != right[2];
    }
    //! Allow inplace addition for BasicVector
    __forceinline__ __device__
    BasicVector<ValueType>& operator+=(const BasicVector<ValueType>& right)
    {
        return *this = *this + right;
    }
    //! Allow inplace subtraction for BasicVector
    __device__
    BasicVector<ValueType>& operator-=(const BasicVector<ValueType>& right)
    {
        return *this = *this - right;
    }
    //! Allow vector addition
    __device__
    BasicVector<ValueType> operator+(const BasicVector<ValueType>& right) const
    {
        return { x_[0] + right[0], x_[1] + right[1], x_[2] + right[2] };
    }
    //! Allow vector subtraction
    __forceinline__ __device__
    BasicVector<ValueType> operator-(const BasicVector<ValueType>& right) const
    {
        return { x_[0] - right[0], x_[1] - right[1], x_[2] - right[2] };
    }
    //! Allow vector scalar division
    __device__
    BasicVector<ValueType> operator/(const ValueType& right) const
    {
        assert((right != 0 && "Cannot divide by zero"));

        return *this * (1 / right);
    }
    //! Scale vector by a scalar
    __device__
    BasicVector<ValueType>& operator*=(const ValueType& right)
    {
        x_[0] *= right;
        x_[1] *= right;
        x_[2] *= right;

        return *this;
    }
    //! Divide vector by a scalar
    __device__
    BasicVector<ValueType>& operator/=(const ValueType& right)
    {
        assert((right != 0 && "Cannot divide by zero"));

        return *this *= 1 / right;
    }
    //! Return dot product
    __forceinline__ __device__
    ValueType dot(const BasicVector<ValueType>& right) const
    {
        return x_[0] * right[0] + x_[1] * right[1] + x_[2] * right[2];
    }

    //! Allow vector vector multiplication (cross product)
    BasicVector<ValueType> cross(const BasicVector<ValueType>& right) const
    {
        return { x_[YY] * right.x_[ZZ] - x_[ZZ] * right.x_[YY],
                 x_[ZZ] * right.x_[XX] - x_[XX] * right.x_[ZZ],
                 x_[XX] * right.x_[YY] - x_[YY] * right.x_[XX] };
    }

    //! Return normalized to unit vector
    BasicVector<ValueType> unitVector() const
    {
        const ValueType vectorNorm = norm();
        assert((vectorNorm != 0 && "unitVector() should not be called with a zero vector"));

        return *this / vectorNorm;
    }

    //! Length^2 of vector
    __forceinline__ __device__
    ValueType norm2() const { return dot(*this); }

    //! Norm or length of vector
    ValueType norm() const { return std::sqrt(norm2()); }

    //! cast to RVec
    BasicVector<real> toRVec() const { return { real(x_[0]), real(x_[1]), real(x_[2]) }; }

    //! cast to IVec
    BasicVector<int> toIVec() const
    {
        return { static_cast<int>(x_[0]), static_cast<int>(x_[1]), static_cast<int>(x_[2]) };
    }

    //! cast to DVec
    BasicVector<double> toDVec() const { return { double(x_[0]), double(x_[1]), double(x_[2]) }; }

    //! Converts to a raw C array where implicit conversion does not work.
    RawArray& as_vec() { return x_; }
    //! Converts to a raw C array where implicit conversion does not work.
    const RawArray& as_vec() const { return x_; }
    //! Makes BasicVector usable in contexts where a raw C array is expected.
    operator RawArray&() { return x_; }
    //! Makes BasicVector usable in contexts where a raw C array is expected.
    operator const RawArray&() const { return x_; }

private:
    RawArray x_;
};

//! Allow vector scalar multiplication
template<typename ValueType>
__forceinline__ __device__
BasicVector<ValueType> operator*(const BasicVector<ValueType>& basicVector, const ValueType& scalar)
{
    return { basicVector[0] * scalar, basicVector[1] * scalar, basicVector[2] * scalar };
}

//! Allow scalar vector multiplication
template<typename ValueType>
__forceinline__ __device__
BasicVector<ValueType> operator*(const ValueType& scalar, const BasicVector<ValueType>& basicVector)
{
    return { scalar * basicVector[0], scalar * basicVector[1], scalar * basicVector[2] };
}

/*! \brief
 * unitv for gmx::BasicVector
 */
template<typename VectorType>
static __forceinline__ VectorType unitVector(const VectorType& v)
{
    return v.unitVector();
}

/*! \brief
 * norm for gmx::BasicVector
 */
template<typename ValueType>
static __forceinline__ ValueType norm(BasicVector<ValueType> v)
{
    return v.norm();
}

/*! \brief
 * Square of the vector norm for gmx::BasicVector
 */
template<typename ValueType>
static __forceinline__ __device__
ValueType norm2(BasicVector<ValueType> v)
{
    return v.norm2();
}

/*! \brief
 * cross product for gmx::BasicVector
 */
template<typename VectorType>
static __forceinline__ VectorType cross(const VectorType& a, const VectorType& b)
{
    return a.cross(b);
}

/*! \brief
 * dot product for gmx::BasicVector
 */
template<typename ValueType>
static __forceinline__ __device__
ValueType dot(BasicVector<ValueType> a, BasicVector<ValueType> b)
{
    return a.dot(b);
}

/*! \brief
 * Multiply two vectors element by element and return the result.
 */
template<typename VectorType>
static __forceinline__ VectorType scaleByVector(const VectorType& a, const VectorType& b)
{
    return { a[0] * b[0], a[1] * b[1], a[2] * b[2] };
}

/*! \brief
 * Return the element-wise minimum of two vectors.
 */
template<typename VectorType>
static __forceinline__ VectorType elementWiseMin(const VectorType& a, const VectorType& b)
{
    return { std::min(a[0], b[0]), std::min(a[1], b[1]), std::min(a[2], b[2]) };
}

/*! \brief
 * Return the element-wise maximum of two vectors.
 */
template<typename VectorType>
static __forceinline__ VectorType elementWiseMax(const VectorType& a, const VectorType& b)
{
    return { std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2]) };
}

/*! \brief
 * Casts a gmx::BasicVector array into an equivalent raw C array.
 */
template<typename ValueType>
static __forceinline__ typename BasicVector<ValueType>::RawArray* as_vec_array(BasicVector<ValueType>* x)
{
    return reinterpret_cast<typename BasicVector<ValueType>::RawArray*>(x);
}

/*! \brief
 * Casts a gmx::BasicVector array into an equivalent raw C array.
 */
template<typename ValueType>
static __forceinline__ const typename BasicVector<ValueType>::RawArray* as_vec_array(const BasicVector<ValueType>* x)
{
    return reinterpret_cast<const typename BasicVector<ValueType>::RawArray*>(x);
}

//! Shorthand for C++ `rvec`-equivalent type.
typedef BasicVector<real> RVec;
//! Shorthand for C++ `dvec`-equivalent type.
typedef BasicVector<double> DVec;
//! Shorthand for C++ `ivec`-equivalent type.
typedef BasicVector<int> IVec;
//! Casts a gmx::RVec array into an `rvec` array.
static __forceinline__ rvec* as_rvec_array(RVec* x)
{
    return as_vec_array(x);
}
//! Casts a gmx::RVec array into an `rvec` array.
static __forceinline__ const rvec* as_rvec_array(const RVec* x)
{
    return as_vec_array(x);
}
//! Casts a gmx::DVec array into an `Dvec` array.
static __forceinline__ dvec* as_dvec_array(DVec* x)
{
    return as_vec_array(x);
}
//! Casts a gmx::IVec array into an `ivec` array.
static __forceinline__ ivec* as_ivec_array(IVec* x)
{
    return as_vec_array(x);
}


//! Casts a gmx::DVec array into an `dvec` array.
static __forceinline__ const dvec* as_dvec_array(const DVec* x)
{
    return as_vec_array(x);
}
//! Casts a gmx::IVec array into an `ivec` array.
static __forceinline__ const ivec* as_ivec_array(const IVec* x)
{
    return as_vec_array(x);
}

//! Shorthand for C++ `ivec`-equivalent type.
typedef BasicVector<int> IVec;

} // namespace gmx

#endif // include guard

