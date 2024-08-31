#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace fs = std::filesystem;

template <typename T>
void print_array(const T *const array, size_t array_length);
template <typename T>
void print_regular_matrix(const T *const *matrix, size_t rows_number, size_t cols_number);
template <typename T>
void print_irregular_matrix(const T *const *matrix, size_t rows_number, const int *const rows_length);
std::vector<fs::path> get_file_paths_in_directory(const fs::path &directory_path);
fs::path select_matrix_file(const std::vector<fs::path> &matrix_paths);
