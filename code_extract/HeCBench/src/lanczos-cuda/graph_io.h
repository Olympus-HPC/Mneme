#ifndef _GRAPH_IO_H
#define _GRAPH_IO_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "matrix.h"

using std::ifstream;
using std::getline;
using std::string;
using std::stringstream;
using std::cerr;
using std::exit;

template <typename T>
coo_matrix<T> adjacency_matrix_from_graph(int node_count, const string &path) {
    coo_matrix<T> matrix(node_count);
    ifstream input_stream(path);
    if (!input_stream.is_open()) {
        cerr << "Error: failed to open the file: " << path << '\n';
        exit(EXIT_FAILURE);
    }
    string line;
    T w(1);
    while (getline(input_stream, line)) {
        stringstream input(line);
        int i, j;
        input >> i >> j;
        if (i >= node_count || j >= node_count) {
            continue;
        }
        matrix.add_entry(i, j, w);
    }
    return matrix;
}

template <typename T>
symm_tridiag_matrix<T> symm_tridiag_matrix_from_file(const string &path) {
    int n;
    ifstream input(path);
    if (!input.is_open()) {
        cerr << "Error: failed to open the file: " << path << '\n';
        exit(EXIT_FAILURE);
    }
    input >> n;
    symm_tridiag_matrix<T> matrix(n);
    for (int i = 0; i < n; ++i) {
        input >> matrix.alpha(i);
    }
    for (int i = 0; i < n - 1; ++i) {
        input >> matrix.beta(i);
    }
    return matrix;
}

#endif
