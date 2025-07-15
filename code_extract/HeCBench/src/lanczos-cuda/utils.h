#ifndef _UTILS_H
#define _UTILS_H

#include <algorithm>
#include <cmath>
#include <iterator>
#include <iostream>
#include <limits>
#include <random>

template <typename T>
void print_vector(const vector<T> &v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

template <typename T>
vector<T> random_vector(int n) {
	std::default_random_engine generator;
	std::uniform_real_distribution<T> distribution(0.0, 1.0);
	vector<T> result(n);
	for (int i = 0; i < n; ++i) {
		result[i] = distribution(generator);
	}
	return result;
}

template <typename T>
T diff_vector(const vector<T> &a, const vector<T> &b) {
	int n = a.size();
	if (n != b.size()) {
		return std::numeric_limits<T>::max();
	}
	T diff(0);
	for (int i = 0; i < n; ++i) {
		diff = std::max(diff, T(std::abs(a[i] - b[i])));
	}
	return diff;
}

#endif
