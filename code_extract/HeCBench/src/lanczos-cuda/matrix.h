#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <cassert>
#include <vector>

using std::vector;

/**
 * @brief A coordinate list sparse matrix.
 * @details
 * 
 * @tparam T Element data type.
 */
template <typename T>
class coo_matrix {
public:
    coo_matrix(int m, int n)
        : row_size_(m), col_size_(n) {}
    coo_matrix(int n) : coo_matrix(n, n) {}

    int row_size() const { return row_size_; }
    int col_size() const { return col_size_; }
    int nonzeros() const { return values_.size(); }

    const T &values(int i) const { return values_[i]; }
    const int &row(int i) const { return row_[i]; }
    const int &col(int i) const { return col_[i]; }

    T &values(int i) { return values_[i]; }
    int &row(int i) { return row_[i]; }
    int &col(int i) { return col_[i]; }

    const T *values_data(int i) const { return values_.data(); }
    const int *row_data(int i) const { return row_.data(); }
    const int *col_data(int i) const { return col_.data(); }

    T *values_data(int i) { return values_.data(); }
    int *row_data(int i) { return row_.data(); }
    int *col_data(int i) { return col_.data(); }

    void add_entry(int i, int j, T &v);

private:
    int row_size_, col_size_;
    vector<T> values_;
    vector<int> row_;
    vector<int> col_;
};

template <typename T>
void coo_matrix<T>::add_entry(int i, int j, T &v) {
    row_.push_back(i);
    col_.push_back(j);
    values_.push_back(v);
}

/**
 * @brief A compressed sparse row (CSR) matrix.
 * @details
 * 
 * @tparam T Element data type.
 */
template <typename T>
class csr_matrix {
public:
    csr_matrix(int m, int n)
        : row_size_(m), col_size_(n), row_ptr_(m + 1, 0) {}
    csr_matrix(int n) : csr_matrix(n, n) {}
    csr_matrix(const coo_matrix<T> &matrix);

    int row_size() const { return row_size_; }
    int col_size() const { return col_size_; }
    int nonzeros() const { return values_.size(); }

    const T &values(int i) const { return values_[i]; }
    const int &col_ind(int i) const { return col_ind_[i]; }
    const int &row_ptr(int i) const { return row_ptr_[i]; }

    const T *values_data() const { return values_.data(); }
    const int *col_ind_data() const { return col_ind_.data(); }
    const int *row_ptr_data() const { return row_ptr_.data(); }

private:
    int row_size_, col_size_;
    vector<T> values_;
    vector<int> col_ind_;
    vector<int> row_ptr_;
};

template <typename T>
csr_matrix<T>::csr_matrix(const coo_matrix<T> &matrix)
    : csr_matrix(matrix.row_size(), matrix.col_size()) {
    for (int i = 0; i < matrix.nonzeros(); ++i) {
        row_ptr_[matrix.row(i)]++;
    }
    for (int i = 1; i < row_size_ + 1; ++i) {
        row_ptr_[i] += row_ptr_[i - 1];
    }
    col_ind_.resize(matrix.nonzeros());
    values_.resize(matrix.nonzeros());
    for (int i = 0; i < matrix.nonzeros(); ++i) {
        int pos = --row_ptr_[matrix.row(i)];
        col_ind_[pos] = matrix.col(i);
        values_[pos] = matrix.values(i);
    }
}

/**
 * @brief A symmetric tridiagonal matrix.
 * @details
 * 
 * @tparam T Element data type.
 */
template <typename T>
class symm_tridiag_matrix {
public:
    symm_tridiag_matrix(int n)
        : alpha_(n), beta_(n - 1) {}

    int size() const { return alpha_.size(); }
    void resize(int n);
    void remove_forward(int i);
    void remove_backward(int i);

    const T &alpha(int i) const { return alpha_[i]; }
    const T &beta(int i) const { return beta_[i]; }

    T &alpha(int i) { return alpha_[i]; }
    T &beta(int i) { return beta_[i]; }

    const T *alpha_data() const { return alpha_.data(); }
    const T *beta_data() const { return beta_.data(); }

    T *alpha_data() { return alpha_.data(); }
    T *beta_data() { return beta_.data(); }

private:
    vector<T> alpha_; /**< main diagonal entries */
    vector<T> beta_; /**< diagonal entries below or above the main diagonal */
};

template <typename T>
void symm_tridiag_matrix<T>::resize(int n) {
    alpha_.resize(n);
    beta_.resize(n - 1);
}

template <typename T>
void symm_tridiag_matrix<T>::remove_forward(int i) {
    assert(i < size() - 1);
    alpha_.erase(alpha_.begin() + i);
    beta_.erase(beta_.begin() + i);
}

template <typename T>
void symm_tridiag_matrix<T>::remove_backward(int i) {
    assert(i > 0);
    alpha_.erase(alpha_.begin() + i);
    beta_.erase(beta_.begin() - 1 + i);
}

#endif
