#pragma once 
#include <random>
#include <vector>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <filesystem>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <XoshiroCpp.hpp>

namespace fs = std::filesystem;

struct H_matrix
{
    int **bit_nodes = nullptr;          // A two-dimensional array that establishes a correspondence between the positions of bit nodes and the positions of check nodes that control them in parity check matrix.
    int *bit_nodes_weight = nullptr;    // It is used in irregular matrices. Determines the number of check nodes that control the current bit node.
    int **check_nodes = nullptr;        // A two-dimensional array that establishes a correspondence between the positions of check nodes and the positions of bit nodes that controlled by them in parity check matrix.
    int *check_nodes_weight = nullptr;  // It is used in irregular matrices. Determines the number of bit nodes that controlled by current check node.
    size_t num_bit_nodes{};             // Number of bit nodes, which is defined as the number of columns in the parity check matrix.
    size_t num_check_nodes{};           // Number of check nodes, which is defined as the number of rows in the parity check matrix.
    size_t max_bit_nodes_weight{};      // Used for regular matrices. All bit nodes have the same number of check nodes.
    size_t max_check_nodes_weight{};    // Used for regular matrices. All check nodes have the same number of bit nodes.
    bool is_regular{};                  // Matrix type.
};

void get_bit_nodes(const std::vector<std::vector<int>> &matrix, const int *const bit_nodes_weight, int **&bit_nodes_out);
void get_check_nodes(const std::vector<std::vector<int>> &matrix, const int *const check_nodes_weight, int **&check_nodes_out);
double get_max_llr_regular(const double *const *matrix, const size_t &nodes_weight, const size_t &rows_number);
double get_max_llr_irregular(const double *const *matrix, const int *const nodes_weight, const size_t &rows_number);
void free_matrix_H(H_matrix &matrix);
bool arrays_equal(const int *const array1, const int *const array2, const size_t &array_length);
void read_sparse_alist_matrix(const fs::path &matrix_path, H_matrix &matrix_out);
void read_dense_matrix(const fs::path &matrix_path, H_matrix &matrix_out);
void generate_random_bit_array(XoshiroCpp::Xoshiro256PlusPlus &prng, size_t length, int *const random_bit_array_out);
double introduce_errors(XoshiroCpp::Xoshiro256PlusPlus &prng, const int *const bit_array, size_t array_length, double error_probability, int *const bit_array_with_errors_out);
void calculate_syndrome_regular(const int *const bit_array, const H_matrix &matrix, int *const syndrome_out);
void calculate_syndrome_irregular(const int *const bit_array, const H_matrix &matrix, int *const syndrome_out);
void threshold_matrix_regular(double *const *matrix, const size_t &rows_number, const size_t &nodes_weight, const double &msg_threshold);
void threshold_matrix_irregular(double *const *matrix, const size_t &rows_number, const int *const nodes_weight, const double &msg_threshold);

// Freeing memory allocated for a two-dimensional dynamic array.
template <typename T>
void free_matrix(T **matrix, const size_t &rows_number)
{
    for (size_t i = 0; i < rows_number; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}














