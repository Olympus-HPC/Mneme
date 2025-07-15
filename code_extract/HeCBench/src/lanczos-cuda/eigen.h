#ifndef _EIGEN_H_
#define _EIGEN_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "cycle_timer.h"
#include "linear_algebra.h"
#include "matrix.h"

using std::cout;
using std::endl;
using std::vector;

/**
 * @brief   Lanczos algorithm for eigendecomposition.
 * 
 * @param   matrix  CSR matrix to decompose
 * @param   k       number of largest eigenvalues to compute
 * @param   steps   maximum steps for the iteration
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> lanczos_eigen(const csr_matrix<T> &matrix, int k, int steps) {
    int n = matrix.row_size();
    assert(n > 0 && n == matrix.col_size());
    assert(steps > 2 * k);

    symm_tridiag_matrix<T> tridiag(steps);
    vector<vector<T> > basis;

    vector<T> r(n, 0);
    r[0] = 1; // initialize a "random" vector
    T beta = l2_norm(r);
    double start_time = cycle_timer::current_seconds();
    for (int t = 0; t < steps; ++t) {
        if (t > 0) {
            tridiag.beta(t - 1) = beta;
        }
        multiply_inplace(r, 1 / beta);
        basis.push_back(r);
        r = multiply(matrix, r);
        T alpha = dot_product(basis[t], r);
        saxpy_inplace(r, -alpha, basis[t]);
        if (t > 0) {
            saxpy_inplace(r, -beta, basis[t - 1]);
        }
        tridiag.alpha(t) = alpha;
        beta = l2_norm(r);
    }
    double end_time = cycle_timer::current_seconds();
    cout << "CPU Lanczos iterations: " << steps << endl;
    cout << "CPU Lanczos time: " << end_time - start_time << " sec" << endl;
    return lanczos_no_spurious(tridiag, k);
}

template <typename T>
vector<T> lanczos_no_spurious(symm_tridiag_matrix<T> &tridiag, int k, const T epsilon = 1e-3) {
    assert(tridiag.size() > 0);
    double start_time = cycle_timer::current_seconds();

    vector<T> eigen = tqlrat_eigen(tridiag);
    tridiag.remove_forward(0);
    vector<T> test_eigen = tqlrat_eigen(tridiag);
    vector<T> result;

    int i = 0;
    int j = 0;
    while (j <= (int)eigen.size()) { // scan through one position beyond the end of the list
        if (j < (int)eigen.size() && std::abs(eigen[j] - eigen[i]) < epsilon) {
            j++;
            continue;
        }
        // simple eigenvalues not in test set are preserved
        // multiple eigenvalues are only preserved once
        if (j - i > 1 || approximate_find(test_eigen, eigen[i], epsilon) == test_eigen.end()) {
            result.push_back(eigen[i]);
        }
        i = j++;
    }
    std::sort(result.rbegin(), result.rend());
    result.resize(std::min((int)result.size(), k));

    double end_time = cycle_timer::current_seconds();
    cout << "spurious removal time: " << end_time - start_time << " sec" << endl;
    return result;
}

/**
 * @brief   Calculating eigenvalues for symmetric tridiagonal matrices.
 * @details Reinsch, C. H. (1973). Algorithm 464: Eigenvalues of a Real, Symmetric, Tridiagonal Matrix.
 *          Communications of the ACM, 16(11), 689.
 * 
 * @param   matrix  symmetric tridiagonal matrix to decompose
 * @param   epsilon precision threshold
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> tqlrat_eigen(const symm_tridiag_matrix<T> &matrix, const T epsilon = 1e-8) {
    double start_time = cycle_timer::current_seconds();

    int n = matrix.size();
    vector<T> d(matrix.alpha_data(), matrix.alpha_data() + n);
    vector<T> e2(n, 0);
    for (int i = 0; i < n - 1; ++i) {
        e2[i] = matrix.beta(i) * matrix.beta(i);
    }
    T b(0), b2(0), f(0);
    for (int k = 0; k < n; ++k) {
        T h = epsilon * epsilon * (d[k] * d[k] + e2[k]);
        if (b2 < h) {
            b = sqrt(h);
            b2 = h;
        }
        int m = k;
        while (m < n && e2[m] > b2) {
            ++m;
        }
        if (m == n) {
            --m;
        }
        if (m > k) {
            do {
                T g = d[k];
                T p2 = sqrt(e2[k]);
                h = (d[k + 1] - g) / (2.0 * p2);
                T r2 = sqrt(h * h + 1.0);
                d[k] = h = p2 / (h < 0.0 ? h - r2 : h + r2);
                h = g - h;
                f = f + h;
                for (int i = k + 1; i < n; ++i) {
                    d[i] -= h;
                }
                h = g = std::abs(d[m] - 0.0) < epsilon ? b : d[m];
                T s2 = 0.0;
                for (int i = m - 1; i >= k; --i) {
                    p2 = g * h;
                    r2 = p2 + e2[i];
                    e2[i + 1] = s2 * r2;
                    s2 = e2[i] / r2;
                    d[i + 1] = h + s2 * (h + d[i]);
                    g = d[i] - e2[i] / g;
                    if (std::abs(g - 0.0) < epsilon) {
                        g = b;
                    }
                    h = g * p2 / r2;
                }
                e2[k] = s2 * g * h;
                d[k] = h;
            } while (e2[k] > b2);
        }
        h = d[k] + f;
        int j;
        for (j = k; j > 0; --j) {
            if (h >= d[j - 1]) {
                break;
            }
            d[j] = d[j - 1];
        }
        d[j] = h;
    }

    double end_time = cycle_timer::current_seconds();
    cout << "TQLRAT time: " << end_time - start_time << " sec" << endl;
    return d;
}

/**
 * @brief   QR eigendecomposition for symmetric tridiagonal matrices.
 * 
 * @param   matrix  symmetric tridiagonal matrix to decompose
 * @param   epsilon precision threshold
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> qr_eigen(const symm_tridiag_matrix<T> &matrix, const T epsilon = 1e-8) {
    double start_time = cycle_timer::current_seconds();
    symm_tridiag_matrix<T> tridiag = matrix;
    int n = tridiag.size();

    tridiag.resize(n + 1);
    tridiag.alpha(n) = 0;
    tridiag.beta(n - 1) = 0;
    for (int i = 0; i < n - 1; ++i) {
        tridiag.beta(i) = tridiag.beta(i) * tridiag.beta(i);
    }
    bool converged = false;
    while (!converged) {
        T diff(0);
        T u(0);
        T ss2(0), s2(0); // previous and current value of s^2
        for (int i = 0; i < n; ++i) {
            T gamma = tridiag.alpha(i) - u;
            T p2 = T(std::abs(1 - s2)) < epsilon ? (1 - ss2) * tridiag.beta(i - 1) : gamma * gamma / (1 - s2);
            if (i > 0) {
                tridiag.beta(i - 1) = s2 * (p2 + tridiag.beta(i));
            }
            ss2 = s2;
            s2 = tridiag.beta(i) / (p2 + tridiag.beta(i));
            u = s2 * (gamma + tridiag.alpha(i + 1));
            // update alpha
            T old = tridiag.alpha(i);
            tridiag.alpha(i) = gamma + u;
            diff = std::max(diff, T(std::abs(old - tridiag.alpha(i))));
        }
        if (diff < epsilon) {
            converged = true;
        }
    }
    double end_time = cycle_timer::current_seconds();
    cout << "QR decomposition time: " << end_time - start_time << " sec" << endl;
    return vector<T>(tridiag.alpha_data(), tridiag.alpha_data() + n);
}

#endif
