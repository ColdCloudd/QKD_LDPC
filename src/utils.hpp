#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <fmt/format.h>

namespace fs = std::filesystem;

void print_array(const int *const array, size_t array_length);
void print_array(const double *const array, size_t array_length);
std::vector<fs::path> get_file_paths_in_directory(const fs::path &directory_path);
fs::path select_matrix_file(const std::vector<fs::path> &matrix_paths);

// Outputs to the console a matrix in which all rows have the same weight.
template <typename T>
void print_regular_matrix(const T *const *matrix, size_t rows_number, size_t cols_number)
{
    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < cols_number; j++)
        {
            fmt::print(fg(fmt::color::blue), "{:.4} ", matrix[i][j]);
        }
        fmt::print("\n");
    }
}

//Outputs to the console a matrix in which some rows may have different weights.
template <typename T>
void print_irregular_matrix(const T *const *matrix, size_t rows_number, const int *const rows_length)
{
    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < rows_length[i]; j++)
        {
            fmt::print(fg(fmt::color::blue), "{:.4} ", matrix[i][j]);
        }
        fmt::print("\n");
    }
}
