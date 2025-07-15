#ifndef _LINEAR_ALGEBRA_H_
#define _LINEAR_ALGEBRA_H_

#include <cassert>
#include <cmath>
#include <vector>

#include "matrix.h"

using std::sqrt;
using std::vector;

template <typename T>
T dot_product(const vector<T> &v1, const vector<T> &v2) {
    int n = v1.size();
    assert(n == v2.size());
    T s = 0;
    for (int i = 0; i < n; ++i) {
        s += v1[i] * v2[i];
    }
    return s;
}

template <typename T>
vector<T> multiply(const csr_matrix<T> &m, const vector<T> &v) {
    int rows = m.row_size();
    int cols = m.col_size();
    assert(cols == v.size());
    vector<T> product(v.size(), 0);
    for (int r = 0; r < rows; ++r) {
        int start = m.row_ptr(r);
        int end = m.row_ptr(r + 1);
        for (int i = start; i < end; ++i) {
            int c = m.col_ind(i);
            product[r] += m.values(i) * v[c];
        }
    }
    return product;
}

template <typename T>
void multiply_inplace(vector<T> &v, const T &k) {
    for (auto p = v.begin(); p != v.end(); ++p) {
        *p *= k;
    }
}

template <typename T>
void add_inplace(vector<T> &v, const T &k) {
    for (auto p = v.begin(); p != v.end(); ++p) {
        *p += k;
    }
}

template <typename T>
void saxpy_inplace(vector<T> &y, const T &a, const vector<T> &x) {
    int n = y.size();
    assert(n == x.size());
    for (int i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

template <typename T>
T l2_norm(const vector<T> &v) {
    return T(sqrt(dot_product(v, v)));
}

template <typename InputIterator, typename T>
InputIterator approximate_find(InputIterator first, InputIterator last, const T &val, const T &eps) {
    while (first != last) {
        if (T(std::abs(*first - val)) < eps) {
            return first;
        }
        ++first;
    }
    return last;
}

template <typename T>
typename vector<T>::const_iterator approximate_find(const vector<T> &input, const T &val, const T &eps) {
    return approximate_find(input.begin(), input.end(), val, eps);
}

#endif
