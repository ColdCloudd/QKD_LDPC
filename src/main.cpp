#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric> 
#include <algorithm>
#include <random>
#include <filesystem>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <BS_thread_pool.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#define DEBUG_SUM_PRODUCT false
#define DEBUG_SUM_PRODUCT_LLR_TRACE false
#define DEBUG_SUM_PRODUCT_MSG_LLR_THRESHOLD false
#define DEBUG_QKD_LDPC false

#define RATE_ADAPTIVE_QBER true
#define INTERACTIVE_MODE true

namespace fs = std::filesystem;

const fs::path DENSE_MATRIX_DIRECTORY_PATH = fs::path(SOURCE_DIR) / "dense_matrices";
const fs::path ALIST_MATRICES_DIRECTORY_PATH = fs::path(SOURCE_DIR) / "alist_sparse_matrices/matrices_PEG_seed_43";
const fs::path ALIST_MATRICES_DIRECTORY_PATH_SIM = fs::path(SOURCE_DIR) / "alist_sparse_matrices/matrices_for_simulations";
const fs::path RESULTS_DIRECTORY_PATH = fs::path(SOURCE_DIR) / "results";

const size_t THREADS_NUMBER = 16;
const size_t SIMULATION_SEED = time(nullptr);
const size_t TRIALS_NUMBER = 100;

const double QBER_START_VALUE = 0.01;
const double QBER_END_VALUE = 0.3;
const double QBER_STEP_VALUE = 0.005;
const size_t MAX_SUM_PRODUCT_ITERATIONS = 100;
const double MESSAGE_THRESHOLD = 100.;

struct H_matrix {
    int** bit_nodes = nullptr;
    int* bit_nodes_weight = nullptr;
    int** check_nodes = nullptr;
    int* check_nodes_weight = nullptr;
    size_t num_bit_nodes{};
    size_t num_check_nodes{};
    size_t max_bit_nodes_weight{};
    size_t max_check_nodes_weight{};
    bool is_regular{};
};

struct sim_input
{
    int sim_number{};
    fs::path matrix_path{};
    std::vector<double> QBER{};
    H_matrix matrix{};
};

struct SP_result
{
    size_t iterations_num{};
    bool syndromes_match{};
};

struct LDPC_result
{
    SP_result sp_res{};
    bool keys_match{};
};

struct trial_result 
{
    LDPC_result ldpc_res{};
    double actual_QBER{};
};

struct sim_result
{
    size_t sim_number{};
    std::string matrix_filename{};
    double code_rate{};
    double actual_QBER{};
    size_t max_iterations_successful_sp{};
    double ratio_trials_successful_sp{};
    double ratio_trials_successful_ldpc{};
};

template <typename T>
void print_array(const T* const array, size_t array_length)
{
    for (size_t i = 0; i < array_length; i++)
    {
        fmt::print(fg(fmt::color::blue), "{} ", array[i]);
    }
}

template <typename T>
void print_regular_matrix(const T* const* matrix, size_t rows_number, size_t cols_number)
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

template <typename T>
void print_irregular_matrix(const T* const* matrix, size_t rows_number, const int* const rows_length)
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

std::vector<fs::path> get_file_paths_in_directory(const fs::path& directory_path) 
{
    std::vector<fs::path> file_paths;
    try 
    {
        if (fs::exists(directory_path) && fs::is_directory(directory_path)) 
        {
            for (const auto& entry : fs::directory_iterator(directory_path)) 
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
    catch (const std::exception& e) 
    {
        fmt::print(stderr, fg(fmt::color::red), "An error occurred while getting file paths in directory: {}\n", directory_path.string());
        throw;
    }

    return file_paths;
}

std::string select_matrix_file(const std::vector<fs::path>& matrix_paths)
{
    std::cout << "Choose file: " << std::endl;
    for (size_t i = 0; i < matrix_paths.size(); i++)
    {
        std::cout << i + 1 << ". " << matrix_paths[i].filename() << std::endl;
    }

    int file_index;
    std::cin >> file_index;
    file_index -= 1;
    if (file_index < 0 || file_index >= static_cast<int>(matrix_paths.size())) {
        std::cout << "Wrong input" << std::endl;
        return "";
    }

    return matrix_paths[file_index].string();
}

bool write_file(const std::vector<sim_result>& data, std::string directory) {
    try
    {
        std::string filename = directory + "qkd_ldpc_cpp(trial_num=" 
            + std::to_string(TRIALS_NUMBER) + "(max_sum_prod_iters=" + std::to_string(MAX_SUM_PRODUCT_ITERATIONS)
            + ",seed=" + std::to_string(SIMULATION_SEED) + ").csv";
        std::fstream fout;
        fout.open(filename, std::ios::out | std::ios::trunc);
        for (size_t i = 0; i < data.size(); i++)
        {
            fout << data[i].sim_number << ";" << data[i].matrix_filename << ";" << data[i].code_rate << ";" << data[i].actual_QBER
                << ";" << data[i].max_iterations_successful_sp << ";" << data[i].ratio_trials_successful_sp << ";" << data[i].ratio_trials_successful_ldpc << "\n";
        }
        fout.close();
        return true;
    }
    catch (const std::exception& ex)
    {
        printf("Error occured (write_file): %s", ex.what());
        return false;
    }
}

std::vector<double> get_QBERs(double start, double end, double step) 
{
    std::vector<double> QBER;
    if (step <= 0) 
    {
        std::cerr << "Error: Step must be positive." << std::endl;
        return QBER;
    }
    else if(start <= 0 || start >= 1 || end <= 0 || end >= 1 || start >= end)
    {
        std::cerr << "Error: Invalid start or end values. Start & end must be in the range [0, 1], and start must be less than end." << std::endl;
        return QBER;
    }
    else if (step > end - start)
    {
        std::cerr << "Error: Step is too large." << std::endl;
        return QBER;
    }

    for (double value = start; value <= end; value += step) 
    {
        QBER.push_back(value);
    }

    return QBER;
}

#if RATE_ADAPTIVE_QBER
std::vector<double> get_rate_adaptive_QBERs(double start, double step, double code_rate)
{
    if (code_rate <= 0.25)
    {
        return get_QBERs(start, 0.04 + step, step);
    }
    else if (code_rate <= 0.5)
    {
        return get_QBERs(start, 0.1 + step, step);
    }
    else if (code_rate <= 0.75)
    {
        return get_QBERs(start, 0.19 + step, step);
    }
    else if (code_rate <= 0.9)
    {
        return get_QBERs(start, 0.25 + step, step);
    }
    else
    {
        return get_QBERs(start, 0.3 + step, step);
    }

}
#endif

void get_bit_nodes(const std::vector<std::vector<int>>& matrix, const int* const bit_nodes_weight, int**& bit_nodes_out)
{
    size_t num_bit_nodes = matrix[0].size();
    size_t num_check_nodes = matrix.size();

    size_t n;
    bit_nodes_out = new int* [num_bit_nodes];
    for (int i = 0; i < num_bit_nodes; i++)
    {
        n = 0;
        bit_nodes_out[i] = new int[bit_nodes_weight[i]];
        for (int j = 0; j < num_check_nodes; j++)
        {
            if (matrix[j][i] == 1)
            {
                bit_nodes_out[i][n] = j;
                n++;
            }
        }
    }
}

void get_check_nodes(const std::vector<std::vector<int>>& matrix, const int* const check_nodes_weight, int**& check_nodes_out)
{
    size_t num_bit_nodes = matrix[0].size();
    size_t num_check_nodes = matrix.size();

    size_t n;
    check_nodes_out = new int* [num_check_nodes];
    for (int i = 0; i < num_check_nodes; i++)
    {
        n = 0;
        check_nodes_out[i] = new int[check_nodes_weight[i]];
        for (int j = 0; j < num_bit_nodes; j++)
        {
            if (matrix[i][j] == 1)
            {
                check_nodes_out[i][n] = j;
                n++;
            }
        }
    }

}

double get_max_llr_regular(const double* const* matrix, const size_t& nodes_weight, const size_t& rows_number)
{
    double max_abs_llr = 0;
    double curr_abs_llr = 0;
    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < nodes_weight; j++)
        {
            curr_abs_llr = abs(matrix[i][j]);
            if (curr_abs_llr > max_abs_llr)
            {
                max_abs_llr = curr_abs_llr;
            }
        }
    }
    return max_abs_llr;
}

double get_max_llr_irregular(const double* const* matrix, const int* const nodes_weight, const size_t& rows_number)
{
    double max_abs_llr = 0;
    double curr_abs_llr = 0;
    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < nodes_weight[i]; j++)
        {
            curr_abs_llr = abs(matrix[i][j]);
            if (curr_abs_llr > max_abs_llr)
            {
                max_abs_llr = curr_abs_llr;
            }
        }
    }
    return max_abs_llr;
}

template <typename T>
void free_matrix(T** matrix, const size_t& rows_number)
{
    for (size_t i = 0; i < rows_number; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void free_matrix_H(H_matrix& matrix)
{
    free_matrix(matrix.bit_nodes, matrix.num_bit_nodes);
    free_matrix(matrix.check_nodes, matrix.num_check_nodes);
    delete[] matrix.bit_nodes_weight;
    delete[] matrix.check_nodes_weight;
}

bool arrays_equal(const int* const array1, const int* const array2, const size_t& array_length) 
{
    for (size_t i = 0; i < array_length; i++)
    {
        if (array1[i] != array2[i])
        {
            return false;
        }
    }
    return true;
}

bool read_alist(const fs::path& matrix_path, H_matrix& matrix_out) {
    std::vector<std::string> line_vec;
    std::ifstream file(matrix_path);

    if (!file.is_open()) 
    {
        std::cerr << "Error: Could not open the file " << matrix_path << std::endl;
        return false;
    }
    std::string line;

    while (getline(file, line)) 
    {
        line_vec.push_back(line);
    }
    file.close();

    if (line_vec.empty()) 
    {
        std::cerr << "Error: File is empty or could not be read properly." << std::endl;
        return false;
    }

    std::vector<std::vector<int>> vec_int;
    try 
    {
        for (const auto& line : line_vec) 
        {
            std::istringstream iss(line);
            std::vector<int> numbers;
            int number;
            while (iss >> number) 
            {
                numbers.push_back(number);
            }
            vec_int.push_back(numbers);
        }
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: An exception occurred while parsing the file: " << e.what() << std::endl;
        return false;
    }

    if (vec_int.size() < 4) {
        std::cerr << "Error: Insufficient data in the file." << std::endl;
        return false;
    }

    if (vec_int[0].size() != 2 || vec_int[1].size() != 2) 
    {
        std::cerr << "Error: Wrong format." << std::endl;
        return false;
    }

    size_t col_num = vec_int[0][0];
    size_t row_num = vec_int[0][1];

    size_t max_col_weight = vec_int[1][0];
    size_t max_row_weight = vec_int[1][1];

    size_t num_bit_nodes = vec_int[2].size();
    size_t num_check_nodes = vec_int[3].size();

    size_t curr_line = 4;

    if (vec_int.size() < curr_line + num_bit_nodes + num_check_nodes)
    {
        std::cerr << "Error: Insufficient data in the file." << std::endl;
        return false;
    }

    if (col_num != num_bit_nodes) 
    {
        std::cerr << "Error: The number of columns (" << col_num << ") is not the same as the length of the third line (" << num_bit_nodes << ")." << std::endl;
        return false;
    }
    else if (row_num != num_check_nodes) 
    {
        std::cerr << "Error: The number of rows (" << row_num << ") is not the same as the length of the fourth line (" << num_check_nodes << ")." << std::endl;
        return false;
    }

    bool is_regular = true;
    matrix_out.bit_nodes_weight = new int[num_bit_nodes];
    matrix_out.check_nodes_weight = new int[num_check_nodes];
    for (size_t i = 0; i < num_bit_nodes; i++)
    {
        matrix_out.bit_nodes_weight[i] = vec_int[2][i];
        if (vec_int[2][i] != vec_int[2][0]) 
        {
            is_regular = false;
        }
    }

    for (size_t i = 0; i < num_check_nodes; i++)
    {
        matrix_out.check_nodes_weight[i] = vec_int[3][i];
        if (vec_int[3][i] != vec_int[3][0])
        {
            is_regular = false;
        }
    }
    
    size_t non_zero_num;
    for (size_t i = 0; i < num_bit_nodes; i++)
    {
        non_zero_num = 0;
        for (size_t j = 0; j < vec_int[curr_line + i].size(); j++)
        {
            if (vec_int[curr_line + i][j] != 0) 
            {
                non_zero_num++;
            }
        }
        if (non_zero_num != vec_int[2][i]) 
        {
            std::cerr << "Error: The number of non-zero elements (" << non_zero_num << ") in the line (" 
                << curr_line + i + 1 << ") does not match the weight in the third line (" << vec_int[2][i] << ")." << std::endl;
            free_matrix_H(matrix_out);
            return false;
        }
    }

    curr_line += num_bit_nodes;
    for (size_t i = 0; i < num_check_nodes; i++)
    {
        non_zero_num = 0;
        for (size_t j = 0; j < vec_int[curr_line + i].size(); j++)
        {
            if (vec_int[curr_line + i][j] != 0)
            {
                non_zero_num++;
            }
        }
        if (non_zero_num != vec_int[3][i])
        {
            std::cerr << "Error: The number of non-zero elements (" << non_zero_num << ") in the line ("
                << curr_line + i + 1 << ") does not match the weight in the fourth line (" << vec_int[3][i] << ")." << std::endl;
            free_matrix_H(matrix_out);
            return false;
        }
    }
    
    try {
        curr_line = 4;
        matrix_out.bit_nodes = new int* [num_bit_nodes];
        for (size_t i = 0; i < num_bit_nodes; ++i)
        {
            matrix_out.bit_nodes[i] = new int[matrix_out.bit_nodes_weight[i]];
            for (size_t j = 0; j < matrix_out.bit_nodes_weight[i]; ++j)
            {
                matrix_out.bit_nodes[i][j] = (vec_int[curr_line + i][j] - 1);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: An exception occurred while creating bit_nodes matrix: " << e.what() << std::endl;
        free_matrix_H(matrix_out);
        return false;
    }

    try {
        curr_line += num_bit_nodes;
        matrix_out.check_nodes = new int* [num_check_nodes];
        for (size_t i = 0; i < num_check_nodes; ++i) 
        {
            matrix_out.check_nodes[i] = new int[matrix_out.check_nodes_weight[i]];
            for (size_t j = 0; j < matrix_out.check_nodes_weight[i]; ++j)
            {
                matrix_out.check_nodes[i][j] = (vec_int[curr_line + i][j] - 1);
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: An exception occurred while creating check_nodes matrix: " << e.what() << std::endl;
        free_matrix_H(matrix_out);
        return false;
    }

    matrix_out.num_check_nodes = row_num;
    matrix_out.num_bit_nodes = col_num;
    matrix_out.max_check_nodes_weight = max_row_weight;
    matrix_out.max_bit_nodes_weight = max_col_weight;
    matrix_out.is_regular = is_regular;

    return true;
}

bool read_raw_matrix(const fs::path& matrix_path, H_matrix& matrix_out) {
    std::vector<std::string> line_vec;
    std::ifstream file(matrix_path);

    if (!file.is_open())
    {
        std::cerr << "Error: Could not open the file " << matrix_path << std::endl;
        return false;
    }
    std::string line;

    while (getline(file, line))
    {
        line_vec.push_back(line);
    }
    file.close();

    if (line_vec.empty())
    {
        std::cerr << "Error: File is empty or could not be read properly." << std::endl;
        return false;
    }

    std::vector<std::vector<int>> vec_int;
    try
    {
        for (const auto& line : line_vec)
        {
            std::istringstream iss(line);
            std::vector<int> numbers;
            int number;
            while (iss >> number)
            {
                if (number != 0 && number != 1) 
                {
                    std::cerr << "Error: The parity check matrix can only take values ​​0 or 1." << std::endl;
                    return false;
                }
                numbers.push_back(number);
            }
            vec_int.push_back(numbers);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: An exception occurred while parsing the file: " << e.what() << std::endl;
        return false;
    }

    for (size_t i = 0; i < vec_int.size(); i++)
    {
        if (vec_int[0].size() != vec_int[i].size()) 
        {
            std::cerr << "Error: Different lengths of rows in a matrix. " << std::endl;
            return false;
        }
    }

    size_t col_num = vec_int[0].size();
    size_t row_num = vec_int.size();

    matrix_out.bit_nodes_weight = new int[col_num];
    matrix_out.check_nodes_weight = new int[row_num];

    size_t curr_weight = 0;
    size_t max_col_weight = 0;
    for (size_t i = 0; i < col_num; i++)
    {
        curr_weight = 0;
        for (size_t j = 0; j < row_num; j++)
        {
            curr_weight += vec_int[j][i];
        }
        if (curr_weight <= 0) 
        {
            std::cerr << "Error: Column (" << i + 1 << ") weight cannot be equal to or less than zero. " << std::endl;
            free_matrix_H(matrix_out);
            return false;
        }
        matrix_out.bit_nodes_weight[i] = curr_weight;
        if (curr_weight > max_col_weight) 
        {
            max_col_weight = curr_weight;
        }
    }

    size_t max_row_weight = 0;
    for (size_t i = 0; i < row_num; i++)
    {
        curr_weight = accumulate(vec_int[i].begin(), vec_int[i].end(), 0);
        if (curr_weight <= 0)
        {
            std::cerr << "Error: Row (" << i + 1 << ") weight cannot be equal to or less than zero. " << std::endl;
            free_matrix_H(matrix_out);
            return false;
        }
        matrix_out.check_nodes_weight[i] = curr_weight;
        if (curr_weight > max_row_weight)
        {
            max_row_weight = curr_weight;
        }
    }

    bool is_regular = true;
    for (size_t i = 0; i < col_num; i++)
    {
        if (matrix_out.bit_nodes_weight[0] != matrix_out.bit_nodes_weight[i])
        {
            is_regular = false;
        }
    }

    for (size_t i = 0; i < row_num; i++)
    {
        if (matrix_out.check_nodes_weight[0] != matrix_out.check_nodes_weight[i])
        {
            is_regular = false;
        }
    }

    get_bit_nodes(vec_int, matrix_out.bit_nodes_weight, matrix_out.bit_nodes);
    get_check_nodes(vec_int, matrix_out.check_nodes_weight, matrix_out.check_nodes);

    matrix_out.num_check_nodes = row_num;
    matrix_out.num_bit_nodes = col_num;
    matrix_out.max_check_nodes_weight = max_row_weight;
    matrix_out.max_bit_nodes_weight = max_col_weight;
    matrix_out.is_regular = is_regular;

    return true;
}

// Generates Alice's key
void generate_random_bit_array(std::mt19937& prng, size_t length, int* const random_bit_array_out) 
{
    std::uniform_int_distribution<int> distribution(0, 1);
    for (int i = 0; i < length; ++i) {
        random_bit_array_out[i] = distribution(prng);
    }
}

// Generates Bob's key by making errors in Alice's key with a given QBER probability (Uniform distribution)
double introduce_errors(std::mt19937& prng, const int* const bit_array, size_t array_length, double error_probability, int* const bit_array_with_errors_out) {
    size_t num_errors = static_cast<size_t>(array_length * error_probability);
    if (num_errors == 0)
    {
        std::copy(bit_array, bit_array + array_length, bit_array_with_errors_out);
    }
    else
    {
        size_t* error_positions = new size_t[array_length];
        for (size_t i = 0; i < array_length; ++i)
        {
            error_positions[i] = i;
        }

        shuffle(error_positions, error_positions + array_length, prng);
        std::copy(bit_array, bit_array + array_length, bit_array_with_errors_out);

        for (size_t i = 0; i < num_errors; ++i)
        {
            bit_array_with_errors_out[error_positions[i]] ^= 1;
        }

        delete[] error_positions;
    }
    return static_cast<double>(num_errors) / array_length;
}

void calculate_syndrome_regular(const int* const bit_array, const H_matrix& matrix, int* const syndrome_out)
{
    std::fill(syndrome_out, syndrome_out + matrix.num_check_nodes, 0);
    for (size_t i = 0; i < matrix.num_check_nodes; i++)
    {
        for (size_t j = 0; j < matrix.max_check_nodes_weight; j++)
        {
            syndrome_out[i] ^= bit_array[matrix.check_nodes[i][j]];
        }
    }
}

void calculate_syndrome_irregular(const int* const bit_array, const H_matrix& matrix, int* const syndrome_out)
{
    std::fill(syndrome_out, syndrome_out + matrix.num_check_nodes, 0);
    for (size_t i = 0; i < matrix.num_check_nodes; i++)
    {
        for (size_t j = 0; j < matrix.check_nodes_weight[i]; j++)
        {
            syndrome_out[i] ^= bit_array[matrix.check_nodes[i][j]];
        }
    }
}

void threshold_matrix_regular(double* const* matrix, const size_t& rows_number, const size_t& nodes_weight, const double& msg_threshold)
{
    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < nodes_weight; j++)
        {
            if (matrix[i][j] > msg_threshold)
            {
                matrix[i][j] = msg_threshold;
            }
            else if (matrix[i][j] < -msg_threshold)
            {
                matrix[i][j] = -msg_threshold;
            }
        }
    }
}

void threshold_matrix_irregular(double* const* matrix, const size_t& rows_number, const int* const nodes_weight, const double& msg_threshold)
{
    for (size_t i = 0; i < rows_number; i++)
    {
        for (size_t j = 0; j < nodes_weight[i]; j++)
        {
            if (matrix[i][j] > msg_threshold)
            {
               matrix[i][j] = msg_threshold;
            }
            else if (matrix[i][j] < -msg_threshold)
            {
                matrix[i][j] = -msg_threshold;
            }
        }
    }
}

SP_result sum_product_decoding_regular(const double* const bit_array_llr, const H_matrix& matrix, const int* const syndrome,
    const size_t& max_num_iterations, const double& msg_threshold, int* const bit_array_out)
{
    #if DEBUG_SUM_PRODUCT_LLR_TRACE
    double max_llr_c2b;
    double max_llr_b2c;
    double max_llr = 0.;
    #endif

    double** bit_to_check_msg = new double* [matrix.num_check_nodes];
    for (size_t i = 0; i < matrix.num_check_nodes; i++)
    {
        bit_to_check_msg[i] = new double[matrix.max_check_nodes_weight];
        for (size_t j = 0; j < matrix.max_check_nodes_weight; j++)
        {
            bit_to_check_msg[i][j] = bit_array_llr[matrix.check_nodes[i][j]];  // Initialization 
        }
    }

    double** check_to_bit_msg = new double* [matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        check_to_bit_msg[i] = new double[matrix.max_bit_nodes_weight];
    }

    double prod;
    double row_prod;
    int curr_bit_pos;

    int* check_pos_idx = new int[matrix.num_bit_nodes];
    double* total_bit_llr = new double[matrix.num_bit_nodes];
    int* decision_syndrome = new int[matrix.num_check_nodes];
    int* bit_pos_idx = new int[matrix.num_check_nodes];

    double sum;
    double col_sum;
    int curr_check_pos;

    size_t curr_iteration = 0;
    while (curr_iteration != max_num_iterations)
    {
    #if DEBUG_SUM_PRODUCT
        cout << "Iteration: " << curr_iteration + 1 << endl;
    #endif

        // # Compute extrinsic messages from check nodes to bit nodes (Step 1: Check messages)
        for (size_t i = 0; i < matrix.num_check_nodes; i++)
        {
            for (size_t j = 0; j < matrix.max_check_nodes_weight; j++)
            {
                bit_to_check_msg[i][j] = tanh(bit_to_check_msg[i][j] / 2.);
            }
        }

        std::fill(check_pos_idx, check_pos_idx + matrix.num_bit_nodes, 0);
        for (size_t j = 0; j < matrix.num_check_nodes; j++)
        {
            row_prod = (syndrome[j]) ? -1. : 1.;
            for (size_t i = 0; i < matrix.max_check_nodes_weight; i++)
            {
                row_prod *= bit_to_check_msg[j][i];
            }

            for (size_t i = 0; i < matrix.max_check_nodes_weight; i++)
            {
                prod = row_prod / bit_to_check_msg[j][i];
                curr_bit_pos = matrix.check_nodes[j][i];
                check_to_bit_msg[curr_bit_pos][check_pos_idx[curr_bit_pos]] = 2. * atanh(prod);
                check_pos_idx[curr_bit_pos]++;
            }
        }

    #if DEBUG_SUM_PRODUCT_MSG_LLR_THRESHOLD
        threshold_matrix_regular(check_to_bit_msg, matrix.num_bit_nodes, matrix.max_bit_nodes_weight, msg_threshold);
    #endif

    #if DEBUG_SUM_PRODUCT
        cout << "E: " << endl;
        print_regular_matrix(check_to_bit_msg, matrix.num_bit_nodes, matrix.max_bit_nodes_weight);
    #endif

        for (size_t i = 0; i < matrix.num_bit_nodes; i++)
        {
            total_bit_llr[i] = std::accumulate(check_to_bit_msg[i], check_to_bit_msg[i] + matrix.max_bit_nodes_weight, bit_array_llr[i]);
            if (total_bit_llr[i] <= 0)
            {
                bit_array_out[i] = 1;
            }
            else
            {
                bit_array_out[i] = 0;
            }
        }

    #if DEBUG_SUM_PRODUCT
        cout << "L: " << endl;
        print_array(total_bit_llr, matrix.num_bit_nodes);
        cout << "z: " << endl;
        print_array(bit_array_out, matrix.num_bit_nodes);
    #endif

        calculate_syndrome_regular(bit_array_out, matrix, decision_syndrome);

    #if DEBUG_SUM_PRODUCT
        cout << "s: " << endl;
        print_array(decision_syndrome, matrix.num_check_nodes);
    #endif

        if (arrays_equal(syndrome, decision_syndrome, matrix.num_check_nodes))
        {
    #if DEBUG_SUM_PRODUCT_LLR_TRACE
            cout << "MAX_LLR = " << max_llr << endl;
    #endif
            free_matrix(bit_to_check_msg, matrix.num_check_nodes);
            free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
            delete[] bit_pos_idx;
            delete[] check_pos_idx;
            delete[] total_bit_llr;
            delete[] decision_syndrome;
            return { curr_iteration + 1, true};
        }

        std::fill(bit_pos_idx, bit_pos_idx + matrix.num_check_nodes, 0);
        for (size_t i = 0; i < matrix.num_bit_nodes; i++)
        {
            col_sum = total_bit_llr[i];
            for (size_t j = 0; j < matrix.max_bit_nodes_weight; j++)
            {
                sum = col_sum - check_to_bit_msg[i][j];
                curr_check_pos = matrix.bit_nodes[i][j];
                bit_to_check_msg[curr_check_pos][bit_pos_idx[curr_check_pos]] = sum;
                bit_pos_idx[curr_check_pos]++;
            }
        }

    #if DEBUG_SUM_PRODUCT_MSG_LLR_THRESHOLD
        threshold_matrix_regular(bit_to_check_msg, matrix.num_check_nodes, matrix.max_check_nodes_weight, msg_threshold);
    #endif

    #if DEBUG_SUM_PRODUCT
        cout << "M: " << endl;
        print_regular_matrix(bit_to_check_msg, matrix.num_check_nodes, matrix.max_check_nodes_weight);
    #endif

    #if DEBUG_SUM_PRODUCT_LLR_TRACE
        max_llr_c2b = get_max_llr_regular(check_to_bit_msg, matrix.max_bit_nodes_weight, matrix.num_bit_nodes);
        max_llr_b2c = get_max_llr_regular(bit_to_check_msg, matrix.max_check_nodes_weight, matrix.num_check_nodes);
        max_llr = max({ max_llr, max_llr_c2b, max_llr_b2c });
    #endif

        curr_iteration++;
    }

    #if DEBUG_SUM_PRODUCT_LLR_TRACE
    cout << "MAX_LLR = " << max_llr << endl;
    #endif

    free_matrix(bit_to_check_msg, matrix.num_check_nodes);
    free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
    delete[] bit_pos_idx;
    delete[] check_pos_idx;
    delete[] total_bit_llr;
    delete[] decision_syndrome;

    return { max_num_iterations, false};
}

SP_result sum_product_decoding_irregular(const double* const bit_array_llr, const H_matrix& matrix, const int* const syndrome, 
    const size_t& max_num_iterations, const double& msg_threshold, int* const bit_array_out)
{
    #if DEBUG_SUM_PRODUCT_LLR_TRACE
    double max_llr_c2b;
    double max_llr_b2c;
    double max_llr = 0.;
    #endif

    double** bit_to_check_msg = new double* [matrix.num_check_nodes];
    for (size_t i = 0; i < matrix.num_check_nodes; i++)
    {
        bit_to_check_msg[i] = new double[matrix.check_nodes_weight[i]];
        for (size_t j = 0; j < matrix.check_nodes_weight[i]; j++)
        {
            bit_to_check_msg[i][j] = bit_array_llr[matrix.check_nodes[i][j]];  // Initialization 
        }
    }

    double** check_to_bit_msg = new double* [matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        check_to_bit_msg[i] = new double[matrix.bit_nodes_weight[i]];
    }

    double prod;
    double row_prod;
    int curr_bit_pos;

    int* check_pos_idx = new int[matrix.num_bit_nodes];
    double* total_bit_llr = new double[matrix.num_bit_nodes];
    int* decision_syndrome = new int[matrix.num_check_nodes];
    int* bit_pos_idx = new int[matrix.num_check_nodes];

    double sum;
    double col_sum;
    int curr_check_pos;

    size_t curr_iteration = 0;
    while (curr_iteration != max_num_iterations)
    {
        #if DEBUG_SUM_PRODUCT
        cout << "Iteration: " << curr_iteration + 1 << endl;
        #endif

        // # Compute extrinsic messages from check nodes to bit nodes (Step 1: Check messages)
        for (size_t i = 0; i < matrix.num_check_nodes; i++)
        {
            for (size_t j = 0; j < matrix.check_nodes_weight[i]; j++)
            {
                bit_to_check_msg[i][j] = tanh(bit_to_check_msg[i][j] / 2.);
            }
        }

        std::fill(check_pos_idx, check_pos_idx + matrix.num_bit_nodes, 0);
        for (size_t j = 0; j < matrix.num_check_nodes; j++)
        {
            row_prod = (syndrome[j]) ? -1. : 1.;
            for (size_t i = 0; i < matrix.check_nodes_weight[j]; i++)
            {
                row_prod *= bit_to_check_msg[j][i];
            }

            for (size_t i = 0; i < matrix.check_nodes_weight[j]; i++)
            {
                prod = row_prod / bit_to_check_msg[j][i];
                curr_bit_pos = matrix.check_nodes[j][i];
                check_to_bit_msg[curr_bit_pos][check_pos_idx[curr_bit_pos]] = 2. * atanh(prod);
                check_pos_idx[curr_bit_pos]++;
            }
        }

        #if DEBUG_SUM_PRODUCT_MSG_LLR_THRESHOLD
        threshold_matrix_irregular(check_to_bit_msg, matrix.num_bit_nodes, matrix.bit_nodes_weight, msg_threshold);
        #endif

        #if DEBUG_SUM_PRODUCT
        cout << "E: " << endl;
        print_irregular_matrix(check_to_bit_msg, matrix.num_bit_nodes, matrix.bit_nodes_weight);
        #endif

        for (size_t i = 0; i < matrix.num_bit_nodes; i++)
        {
            total_bit_llr[i] = std::accumulate(check_to_bit_msg[i], check_to_bit_msg[i] + matrix.bit_nodes_weight[i], bit_array_llr[i]);
            if (total_bit_llr[i] <= 0)
            {
                bit_array_out[i] = 1;
            }
            else
            {
                bit_array_out[i] = 0;
            }
        }

        #if DEBUG_SUM_PRODUCT
        cout << "L: " << endl;
        print_array(total_bit_llr, matrix.num_bit_nodes);
        cout << "z: " << endl;
        print_array(bit_array_out, matrix.num_bit_nodes);
        #endif

        calculate_syndrome_irregular(bit_array_out, matrix, decision_syndrome);

        #if DEBUG_SUM_PRODUCT
        cout << "s: " << endl;
        print_array(decision_syndrome, matrix.num_check_nodes);
        #endif

        if (arrays_equal(syndrome, decision_syndrome, matrix.num_check_nodes))
        {
            #if DEBUG_SUM_PRODUCT_LLR_TRACE
            cout << "MAX_LLR = " << max_llr << endl;
            #endif
            free_matrix(bit_to_check_msg, matrix.num_check_nodes);
            free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
            delete[] bit_pos_idx;
            delete[] check_pos_idx;
            delete[] total_bit_llr;
            delete[] decision_syndrome;
            return { curr_iteration + 1, true};
        }

        std::fill(bit_pos_idx, bit_pos_idx + matrix.num_check_nodes, 0);
        for (size_t i = 0; i < matrix.num_bit_nodes; i++)
        {
            col_sum = total_bit_llr[i];
            for (size_t j = 0; j < matrix.bit_nodes_weight[i]; j++)
            {
                sum = col_sum - check_to_bit_msg[i][j];
                curr_check_pos = matrix.bit_nodes[i][j];
                bit_to_check_msg[curr_check_pos][bit_pos_idx[curr_check_pos]] = sum;
                bit_pos_idx[curr_check_pos]++;
            }
        }

        #if DEBUG_SUM_PRODUCT_MSG_LLR_THRESHOLD
        threshold_matrix_irregular(bit_to_check_msg, matrix.num_check_nodes, matrix.check_nodes_weight, msg_threshold);
        #endif

        #if DEBUG_SUM_PRODUCT
        cout << "M: " << endl;
        print_irregular_matrix(bit_to_check_msg, matrix.num_check_nodes, matrix.check_nodes_weight);
        #endif

        #if DEBUG_SUM_PRODUCT_LLR_TRACE
        max_llr_c2b = get_max_llr_irregular(check_to_bit_msg, matrix.bit_nodes_weight, matrix.num_bit_nodes);
        max_llr_b2c = get_max_llr_irregular(bit_to_check_msg, matrix.check_nodes_weight, matrix.num_check_nodes);
        max_llr = max({ max_llr, max_llr_c2b, max_llr_b2c });
        #endif
        
        curr_iteration++;
    }

    #if DEBUG_SUM_PRODUCT_LLR_TRACE
    cout << "MAX_LLR = " << max_llr << endl;
    #endif

    free_matrix(bit_to_check_msg, matrix.num_check_nodes);
    free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
    delete[] bit_pos_idx;
    delete[] check_pos_idx;
    delete[] total_bit_llr;
    delete[] decision_syndrome;

    return { max_num_iterations, false};
}

LDPC_result QKD_LDPC_regular(const int* const alice_bit_array, const int* const bob_bit_array, const double& QBER, const H_matrix& matrix)
{
    double log_p = log((1. - QBER) / QBER);
    double* apriori_llr = new double[matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        apriori_llr[i] = (bob_bit_array[i] ? -log_p : log_p);
    }

    #if DEBUG_QKD_LDPC
    cout << "r: " << endl;
    print_array(apriori_llr, matrix.num_bit_nodes);
    #endif

    int* alice_syndrome = new int[matrix.num_check_nodes];
    calculate_syndrome_regular(alice_bit_array, matrix, alice_syndrome);

    #if DEBUG_QKD_LDPC
    cout << "Alice syndrome: " << endl;
    print_array(alice_syndrome, matrix.num_check_nodes);
    #endif

    int* bob_solution = new int[matrix.num_bit_nodes];
    LDPC_result ldpc_res;
    ldpc_res.sp_res = sum_product_decoding_regular(apriori_llr, matrix, alice_syndrome, MAX_SUM_PRODUCT_ITERATIONS,
        MESSAGE_THRESHOLD, bob_solution);

    #if DEBUG_QKD_LDPC
    cout << "Bob corrected bit array: " << endl;
    print_array(bob_solution, matrix.num_bit_nodes);
    #endif

    ldpc_res.keys_match = arrays_equal(alice_bit_array, bob_solution, matrix.num_bit_nodes);

    #if DEBUG_QKD_LDPC
    cout << "Iterations performed: " << ldpc_res.sp_res.iterations_num << endl;
    cout << "Syndromes are match: " << ((ldpc_res.sp_res.syndromes_match) ? "YES" : "NO") << endl;
    cout << "Keys are match: " << ((ldpc_res.keys_match) ? "YES" : "NO") << endl;
    #endif

    delete[] apriori_llr;
    delete[] alice_syndrome;
    delete[] bob_solution;

    return ldpc_res;
}

LDPC_result QKD_LDPC_irregular(const int* const alice_bit_array, const int* const bob_bit_array, const double& QBER, const H_matrix& matrix)
{
    double log_p = log((1. - QBER) / QBER);
    double* apriori_llr = new double[matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        apriori_llr[i] = (bob_bit_array[i] ? -log_p : log_p);
    }

    #if DEBUG_QKD_LDPC
    cout << "r: " << endl;
    print_array(apriori_llr, matrix.num_bit_nodes);
    #endif

    int* alice_syndrome = new int[matrix.num_check_nodes];
    calculate_syndrome_irregular(alice_bit_array, matrix, alice_syndrome);
    
    #if DEBUG_QKD_LDPC
    cout << "Alice syndrome: " << endl;
    print_array(alice_syndrome, matrix.num_check_nodes);
    #endif

    int* bob_solution = new int[matrix.num_bit_nodes];
    LDPC_result ldpc_res;
    ldpc_res.sp_res = sum_product_decoding_irregular(apriori_llr, matrix, alice_syndrome, MAX_SUM_PRODUCT_ITERATIONS,
        MESSAGE_THRESHOLD, bob_solution);

    #if DEBUG_QKD_LDPC
    cout << "Bob corrected bit array: " << endl;
    print_array(bob_solution, matrix.num_bit_nodes);
    #endif

    ldpc_res.keys_match = arrays_equal(alice_bit_array, bob_solution, matrix.num_bit_nodes);

    #if DEBUG_QKD_LDPC
    cout << "Iterations performed: " << ldpc_res.sp_res.iterations_num << endl;
    cout << "Syndromes are match: " << ((ldpc_res.sp_res.syndromes_match) ? "YES" : "NO") << endl;
    cout << "Keys are match: " << ((ldpc_res.keys_match) ? "YES" : "NO") << endl;
    #endif

    delete[] apriori_llr;
    delete[] alice_syndrome;
    delete[] bob_solution; 

    return ldpc_res;
}

bool QKD_LDPC_interactive_simulation(std::string alist_mat_dir_path, double QBER_start, double QBER_end, double QBER_step)
{
    H_matrix matrix;
    std::vector<fs::path> matrix_paths = get_file_paths_in_directory(alist_mat_dir_path);
    std::string matrix_path = select_matrix_file(matrix_paths);
    if (matrix_path.empty())
    {
        std::cerr << "Error: Matrix path is empty." << std::endl;
        return false;
    }

    bool is_ready = read_alist(matrix_path, matrix);
    if (!is_ready)
    {
        return false;
    }

    std::cout << ((matrix.is_regular) ? "\nMatrix H is regular.\n" : "\nMatrix H is irregular.\n") << std::endl;
    size_t num_check_nodes = matrix.num_check_nodes;
    size_t num_bit_nodes = matrix.num_bit_nodes;
    int* alice_bit_array = new int[num_bit_nodes];
    int* bob_bit_array = new int[num_bit_nodes];

    std::mt19937 prng(SIMULATION_SEED);
    std::vector<double> QBER = get_QBERs(QBER_start, QBER_end, QBER_step);
    for (size_t i = 0; i < QBER.size(); i++)
    {
        generate_random_bit_array(prng, num_bit_nodes, alice_bit_array);
        double actual_QBER = introduce_errors(prng, alice_bit_array, num_bit_nodes, QBER[i], bob_bit_array);

        std::cout << "Actual QBER: " << actual_QBER << std::endl;
        if (actual_QBER == 0.)
        {
            std::cerr << "Error: Array size is too small for QBER." << std::endl;
            free_matrix_H(matrix);
            delete[] alice_bit_array;
            delete[] bob_bit_array;
            return false;
        }

        int error_num = 0;
        for (size_t i = 0; i < num_bit_nodes; i++)
        {
            error_num += alice_bit_array[i] ^ bob_bit_array[i];
        }
        std::cout << "Errors number: " << error_num << std::endl;

        LDPC_result try_result;
        if (matrix.is_regular)
        {
            try_result = QKD_LDPC_regular(alice_bit_array, bob_bit_array, actual_QBER, matrix);
        }
        else
        {
            try_result = QKD_LDPC_irregular(alice_bit_array, bob_bit_array, actual_QBER, matrix);
        }
        std::cout << "Iterations performed: " << try_result.sp_res.iterations_num << std::endl;
        std::cout << ((try_result.keys_match && try_result.sp_res.syndromes_match) ? "Error reconciliation SUCCESSFUL\n" : "Error reconciliation FAILED\n") << std::endl;
    }

    free_matrix_H(matrix);
    delete[] alice_bit_array;
    delete[] bob_bit_array;
    return true;
}

bool prepare_sim_inputs(const std::vector<fs::path>& matrix_paths, double QBER_start, double QBER_end, double QBER_step, std::vector<sim_input>& sim_inputs_out)
{
    bool is_ready = true;
    for (size_t i = 0; i < matrix_paths.size(); i++)
    {
        if (!read_alist(matrix_paths[i], sim_inputs_out[i].matrix))
        {
           std::cerr << "Error: Matrix " + matrix_paths[i].filename().string() + " is corrupted!" << std::endl;
            is_ready = false;
            break;
        }
        sim_inputs_out[i].sim_number = i;
        sim_inputs_out[i].matrix_path = matrix_paths[i];

        #if RATE_ADAPTIVE_QBER
        double rate = static_cast<double>(sim_inputs_out[i].matrix.num_check_nodes) / sim_inputs_out[i].matrix.num_bit_nodes;
        sim_inputs_out[i].QBER = get_rate_adaptive_QBERs(QBER_start, QBER_step, rate);
        #else
        sim_inputs_out[i].QBER = get_QBERs(QBER_start, QBER_end, QBER_step);
        #endif
    }
    return is_ready;
}

trial_result run_trial(const H_matrix& matrix, double QBER, size_t seed)
{
    std::mt19937 prng(seed);

    trial_result result;
    int* alice_bit_array = new int[matrix.num_bit_nodes];
    int* bob_bit_array = new int[matrix.num_bit_nodes];
    generate_random_bit_array(prng, matrix.num_bit_nodes, alice_bit_array);
    result.actual_QBER = introduce_errors(prng, alice_bit_array, matrix.num_bit_nodes, QBER, bob_bit_array);
    if (result.actual_QBER == 0.)
    {
        return result;
    }

    if (matrix.is_regular)
    {
        result.ldpc_res = QKD_LDPC_regular(alice_bit_array, bob_bit_array, result.actual_QBER, matrix);
    }
    else
    {
        result.ldpc_res = QKD_LDPC_irregular(alice_bit_array, bob_bit_array, result.actual_QBER, matrix);
    }
    delete[] alice_bit_array;
    delete[] bob_bit_array;

    return result;
}

std::vector<sim_result> QKD_LDPC_auto_simulation(const std::vector<sim_input>& sim_in)
{
    using namespace indicators;
    size_t sim_total = 0;
    for (size_t i = 0; i < sim_in.size(); i++)
    {
        sim_total += sim_in[i].QBER.size();
    }

    size_t trials_total = sim_total * TRIALS_NUMBER;
    indicators::show_console_cursor(false);
    indicators::ProgressBar bar{
      option::BarWidth{50},
      option::Start{" ["},
      option::Fill{"="},
      option::Lead{">"},
      option::Remainder{"-"},
      option::End{"]"},
      option::PrefixText{"PROGRESS"},
      option::ForegroundColor{Color::green},
      option::ShowElapsedTime{true},
      option::ShowRemainingTime{true},
      option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
      option::MaxProgress{trials_total}
    };

    size_t curr_sim = 0;
    size_t iteration = 0;
    std::vector<sim_result> sim_results(sim_total);
    std::vector<trial_result> trial_results(TRIALS_NUMBER);

    std::mt19937 prng(SIMULATION_SEED);
    std::uniform_int_distribution<size_t> distribution(0, std::numeric_limits<size_t>::max());

    BS::thread_pool pool(THREADS_NUMBER);
    for (size_t i = 0; i < sim_in.size(); i++)
    {
        const H_matrix& matrix = sim_in[i].matrix;
        double code_rate = static_cast<double>(matrix.num_check_nodes) / matrix.num_bit_nodes;
        std::string matrix_filename = sim_in[i].matrix_path.filename().string();
        for (size_t j = 0; j < sim_in[i].QBER.size(); j++)
        {
            iteration += TRIALS_NUMBER;
            bar.set_option(option::PostfixText{
                        std::to_string(iteration) + "/" + std::to_string(trials_total)
                });

            double QBER = sim_in[i].QBER[j];
            pool.detach_loop<size_t>(0, TRIALS_NUMBER,
                [&matrix, &QBER, &trial_results, &prng, &distribution, &bar](size_t k)
                {
                    trial_results[k] = run_trial(matrix, QBER, distribution(prng));
                    bar.tick();     // for correct time estimation
                });
            pool.wait();

            size_t trials_successful_sp = 0;
            size_t trials_successful_ldpc = 0;
            size_t max_iterations_successful_sp = 0;
            size_t curr_sp_iterations_num;
            for (size_t k = 0; k < trial_results.size(); k++)
            {
                if (trial_results[k].ldpc_res.sp_res.syndromes_match)
                {
                    trials_successful_sp++;
                    curr_sp_iterations_num = trial_results[k].ldpc_res.sp_res.iterations_num;
                    if (max_iterations_successful_sp < curr_sp_iterations_num)
                    {
                        max_iterations_successful_sp = curr_sp_iterations_num;
                    }
                    if (trial_results[k].ldpc_res.keys_match)
                    {
                        trials_successful_ldpc++;
                    }
                }
            }

            sim_results[curr_sim].code_rate = code_rate;
            sim_results[curr_sim].sim_number = sim_in[i].sim_number;
            sim_results[curr_sim].matrix_filename = matrix_filename;

            sim_results[curr_sim].actual_QBER = trial_results[0].actual_QBER;
            sim_results[curr_sim].max_iterations_successful_sp = max_iterations_successful_sp;
            sim_results[curr_sim].ratio_trials_successful_ldpc = static_cast<double>(trials_successful_ldpc) / TRIALS_NUMBER;
            sim_results[curr_sim].ratio_trials_successful_sp = static_cast<double>(trials_successful_sp) / TRIALS_NUMBER;
            curr_sim++;
        }
    }
    return sim_results;
}

int main()
{
    #if INTERACTIVE_MODE
    QKD_LDPC_interactive_simulation(ALIST_MATRICES_DIRECTORY_PATH, QBER_START_VALUE, QBER_END_VALUE, QBER_STEP_VALUE);

    #else
    std::vector<fs::path> matrix_paths = get_file_paths_in_directory(ALIST_MATRICES_DIRECTORY_PATH_SIM);
    if (matrix_paths.empty())
    {
        std::cerr << "Error: Matrix folder is empty." << std::endl;
        return false;
    }

    std::vector<sim_input> sim_inputs(matrix_paths.size());
    bool is_ready_to_sim = prepare_sim_inputs(matrix_paths, QBER_START_VALUE, QBER_END_VALUE, QBER_STEP_VALUE, sim_inputs);
    if (!is_ready_to_sim)
    {
        for (size_t i = 0; i < sim_inputs.size(); i++)
        {
            free_matrix_H(sim_inputs[i].matrix);
        }
        sim_inputs.clear();
        std::cout << "Dynamic memory for storing matrices has been successfully freed!" << std::endl;
        return false;
    }
    std::vector<sim_result> sim_results = QKD_LDPC_auto_simulation(sim_inputs);

    for (size_t i = 0; i < sim_inputs.size(); i++)
    {
        free_matrix_H(sim_inputs[i].matrix);
    }
    sim_inputs.clear();
    std::cout << "Dynamic memory for storing matrices has been successfully freed!" << std::endl;

    std::cout << "The results will be written to the directory: " << RESULTS_DIRECTORY_PATH << std::endl;
    if (write_file(sim_results, RESULTS_DIRECTORY_PATH))
    {
        std::cout << "The results were written to the file successfully!" << std::endl;
    }
    else
    {
        std::cerr << "An error occurred while writing to the file" << std::endl;
    }
    #endif
    return true;
}

