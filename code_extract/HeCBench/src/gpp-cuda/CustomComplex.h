//
// Complex number computation class
//
#ifndef __CustomComplex
#define __CustomComplex

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sys/time.h>

#if defined(__NVCC__) || defined(__HIPCC__)
#define ESS __host__ __device__
#else
#define ESS
#endif

#define nstart 0
#define nend 3

template <class T>
class CustomComplex
{
  private:
  public:
    T x;
    T y;
    explicit CustomComplex()
    {
      x = 0.00;
      y = 0.00;
    }

    ESS
      explicit CustomComplex(const T& a, const T& b)
      {
        x = a;
        y = b;
      }

    ESS
      CustomComplex(const CustomComplex& src)
      {
        x = src.x;
        y = src.y;
      }

    ESS
      CustomComplex& operator=(const CustomComplex& src)
      {
        x = src.x;
        y = src.y;
        return *this;
      }

    ESS
      CustomComplex& operator+=(const CustomComplex& src)
      {
        x = src.x + this->x;
        y = src.y + this->y;
        return *this;
      }

    ESS
      CustomComplex& operator-=(const CustomComplex& src)
      {
        x = src.x - this->x;
        y = src.y - this->y;
        return *this;
      }

    ESS
      CustomComplex& operator-()
      {
        x = -this->x;
        y = -this->y;
        return *this;
      }

    ESS
      CustomComplex conj()
      {
        T re_this = this->x;
        T im_this = -1 * this->y;

        CustomComplex<T> result(re_this, im_this);
        return result;
      }

    ESS
      T real() { return this->x; }

    ESS
      T imag() { return this->y; }

    ESS
      CustomComplex& operator~() { return *this; }

    void print() const
    {
      printf("( %f, %f) ", this->x, this->y);
      printf("\n");
    }

    friend std::ostream& operator<<(std::ostream& os, const CustomComplex<T>& obj)
    {
      os << "( " << obj.x << ", " << obj.y << ") ";
      return os;
    }

    T get_real() const { return this->x; }

    T get_imag() const { return this->y; }

    void set_real(T val) { this->x = val; }

    void set_imag(T val) { this->y = val; }

    // 6 flops
      ESS 
      friend inline CustomComplex<T> operator*(const CustomComplex<T>& a,
          const CustomComplex<T>& b)
      {
        T                x_this = a.x * b.x - a.y * b.y;
        T                y_this = a.x * b.y + a.y * b.x;
        CustomComplex<T> result(x_this, y_this);
        return (result);
      }

    // 2 flops
      ESS 
      friend inline CustomComplex<T> operator*(const CustomComplex<T> &a,
          const T &b)
      {
        CustomComplex<T> result(a.x * b, a.y * b);
        return result;
      }

      ESS 
      friend inline CustomComplex<T> operator*(const T &b,
          const CustomComplex<T> &a)
      {
        CustomComplex<T> result(a.x * b, a.y * b);
        return result;
      }

      ESS 
      friend inline CustomComplex<T> operator-(const CustomComplex<T> &a,
          const CustomComplex<T> &b)
      {
        CustomComplex<T> result(a.x - b.x, a.y - b.y);
        return result;
      }

    // 2 flops
      ESS 
      friend inline CustomComplex<T> operator-(const T &a,
          const CustomComplex<T> &src)
      {
        CustomComplex<T> result(a - src.x, 0 - src.y);
        return result;
      }

      ESS
      friend inline CustomComplex<T> operator+(const T &a,
          CustomComplex<T> &src)
      {
        CustomComplex<T> result(a + src.x, src.y);
        return result;
      }

      ESS
      friend inline CustomComplex<T> operator+(CustomComplex<T> &a,
          CustomComplex<T> &b)
      {
        CustomComplex<T> result(a.x + b.x, a.y + b.y);
        return result;
      }

      ESS
      friend inline CustomComplex<T> operator/(CustomComplex<T> &a,
          CustomComplex<T> &b)
      {
        CustomComplex<T> b_conj      = CustomComplex_conj(b);
        CustomComplex<T> numerator   = a * b_conj;
        CustomComplex<T> denominator = b * b_conj;

        T re_this = numerator.x / denominator.x;
        T im_this = numerator.y / denominator.x;

        CustomComplex<T> result(re_this, im_this);
        return result;
      }

      ESS
      friend inline CustomComplex<T> operator/(CustomComplex<T> &a, T &b)
      {
        CustomComplex<T> result(a.x / b, a.y / b);
        return result;
      }

      ESS
      friend inline CustomComplex<T> CustomComplex_conj(
          const CustomComplex<T> &src)
      {   
        T re_this = src.x;
        T im_this = -1 * src.y;
       
        CustomComplex<T> result(re_this, im_this);
        return result;
       }

      ESS
      friend inline T CustomComplex_abs(const CustomComplex<T> &src)
      {
        T re_this = src.x * src.x;
        T im_this = src.y * src.y;

        T result = sqrt(re_this + im_this);
        return result;
      }

      ESS
      friend inline T CustomComplex_real(const CustomComplex<T> &src) 
      {
        return src.x;
      }

      ESS
      friend inline T CustomComplex_imag(const CustomComplex<T> &src)
      {
        return src.y;
      }
};
#endif
