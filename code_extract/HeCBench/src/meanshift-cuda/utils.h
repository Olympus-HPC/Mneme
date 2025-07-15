#ifndef UTILS_H
#define UTILS_H

#include <array>
#include <algorithm> // all_of
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace mean_shift::gpu::utils {

  template <const size_t N, const size_t D>
    std::array<float, N * D> load_csv(const std::string& path, const char delim) {
      assert(std::filesystem::exists(path));
      std::ifstream file(path);
      std::string line;
      std::array<float, N * D> data_matrix;
      for (size_t i = 0; i < N; ++i) {
        std::getline(file, line);
        std::stringstream line_stream(line);
        std::string cell;
        for (size_t j = 0; j < D; ++j) {
          std::getline(line_stream, cell, delim);
          data_matrix[i * D + j] = std::stof(cell);
        }
      }
      file.close();
      return data_matrix;
    }

  template <const size_t N, const size_t D>
    void write_csv(const std::array<float, N * D>& data, const std::string& path, const char delim) {
      std::ofstream output(path);
      for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < D - 1; ++j)
          output << data[i * D + j] << delim;
        output << data[i * D + D - 1] << '\n';
      }
      output.close();
      return;
    }

  template <typename T, const size_t K>
    void write_csv(const std::array<T, K>& data, const std::string& path, const char delim) {
      std::ofstream output(path);
      for (size_t i = 0; i < K; ++i) {
        output << data[i] << '\n';
      }
      output.close();
      return;
    }

  template <const size_t N, const size_t D>
    void print_data(const std::array<float, N * D>& data) {
      for (const auto& c : data) {
        for (int i = 0; i < D; i++)
          std::cout << c << ' ';
        std::cout << '\n';
      }
      return;
    }

  template <typename T, const size_t M>
    void print_data(const std::array<T, M>& data) {
      for (const auto& c : data)
        std::cout << c << '\n';
      return;
    }

  template <const size_t D>
    void print_data(const std::vector<std::array<float, D>>& data) {
      for (const auto& c : data) {
        for (int i = 0; i < D; i++)
          std::cout << c[i] << ' ';
        std::cout << '\n';
      }
      return;
    }

  void print_info(const std::string PATH_TO_DATA, 
      const size_t N, 
      const size_t D, 
      const size_t BLOCKS,
      const size_t THREADS) {
    std::cout << "\nDATASET:    " << PATH_TO_DATA << '\n';
    std::cout << "NUM POINTS: " << N << '\n';
    std::cout << "DIM:        " << D << '\n';
    std::cout << "BLOCKS:     " <<  BLOCKS << '\n';
    std::cout << "THREADS:    " << THREADS << '\n';
    return;
  }

  void print_info(const std::string PATH_TO_DATA, 
      const size_t N, 
      const size_t D, 
      const size_t BLOCKS,
      const size_t THREADS,
      const size_t TILE_WIDTH) {
    std::cout << "\nDATASET:    " << PATH_TO_DATA << '\n';
    std::cout << "NUM POINTS: " << N << '\n';
    std::cout << "DIM:        " << D << '\n';
    std::cout << "BLOCKS:     " <<  BLOCKS << '\n';
    std::cout << "THREADS:    " << THREADS << '\n';
    std::cout << "TILE WIDTH: " << TILE_WIDTH << '\n';
    return;
  }

  void swap(float* &a, float* &b){
    float *temp = a;
    a = b;
    b = temp;
    return;
  }

  template <const size_t N, const size_t D>
    std::vector<std::array<float, D>> reduce_to_centroids(std::array<float, N * D>& data, const float min_distance) {
      std::vector<std::array<float, D>> centroids;
      centroids.reserve(4);
      std::array<float, D> first_centroid;
      for (size_t j = 0; j < D; ++j) {
        first_centroid[j] = data[j];
      }
      centroids.emplace_back(first_centroid);
      for (size_t i = 0; i < N; ++i) {
        bool at_least_one_close = false;
        for (const auto& c : centroids) {
          float dist = 0;
          for (size_t j = 0; j < D; ++j) {
            dist += ((data[i * D + j] - c[j])*(data[i * D + j] - c[j]));
          }
          if (dist <= min_distance) {
            at_least_one_close = true;
          }
        }
        if (not at_least_one_close) {
          std::array<float, D> centroid;
          for (size_t j = 0; j < D; ++j) {
            centroid[j] = data[i * D + j];
          }
          centroids.emplace_back(centroid);
        }
      }
      return centroids;
    }

  template <const size_t M, const size_t D>
    bool are_close_to_real(const std::vector<std::array<float, D>>& centroids,
        const std::array<float, M * D>& real,
        const float eps_to_real) {
      std::array<bool, M> are_close {false};
      for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < M; ++j) {
          float dist = 0;
          for (size_t k = 0; k < D; ++k) {
            dist += ((centroids[i][k] - real[j * D + k])*(centroids[i][k] - real[j * D + k]));
          }
          if (dist <= eps_to_real) {
            are_close[i] = true;
          }
        }
      }
      return std::all_of(are_close.begin(), are_close.end(), [](const bool b){return b;}); 
    }
}

#endif
