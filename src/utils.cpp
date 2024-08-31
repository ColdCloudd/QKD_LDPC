#include "utils.hpp"

template <typename T>
void print_array(const T *const array, size_t array_length)
{
    for (size_t i = 0; i < array_length; i++)
    {
        fmt::print(fg(fmt::color::blue), "{} ", array[i]);
    }
}

// Outputs to the console a matrix in which all rows have the same weight.
template <typename T>
void print_regular_matrix(const T *const *matrix, size_t rows_number, size_t cols_number)
{
    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < cols_number; j++)
        {
            fmt::print(fg(fmt::color::blue), "{} ", matrix[i][j]);
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
            fmt::print(fg(fmt::color::blue), "{} ", matrix[i][j]);
        }
        fmt::print("\n");
    }
}

// Gets paths to files in the given directory.
std::vector<fs::path> get_file_paths_in_directory(const fs::path &directory_path)
{
    std::vector<fs::path> file_paths;
    try
    {
        if (fs::exists(directory_path) && fs::is_directory(directory_path))
        {
            for (const auto &entry : fs::directory_iterator(directory_path))
            {
                if (fs::is_regular_file(entry.path()))
                {
                    file_paths.push_back(entry.path());
                }
            }
        }
        else
        {
            throw std::runtime_error("Directory doesn't exist.");
        }
    }
    catch (const std::exception &e)
    {
        fmt::print(stderr, fg(fmt::color::red), "An error occurred while getting file paths in directory: {}\n", directory_path.string());
        throw;
    }

    return file_paths;
}

// Allow the user to select a matrix file from available paths
fs::path select_matrix_file(const std::vector<fs::path> &matrix_paths)
{
    fmt::print(fg(fmt::color::green), "Choose file: \n");
    for (size_t i = 0; i < matrix_paths.size(); i++)
    {
        fmt::print(fg(fmt::color::green), "{0}. {1}\n", i + 1, matrix_paths[i].filename().string());
    }

    int file_index;
    std::cin >> file_index;
    file_index -= 1;
    if (file_index < 0 || file_index >= static_cast<int>(matrix_paths.size()))
    {
        throw std::runtime_error("Wrong file number.");
    }
    return matrix_paths[file_index];
}