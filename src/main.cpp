﻿#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <BS_thread_pool.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

const fs::path CONFIG_PATH = fs::path(SOURCE_DIR) / "config.json";
const fs::path DENSE_MATRIX_DIR_PATH = fs::path(SOURCE_DIR) / "dense_matrices";
const fs::path ALIST_MATRIX_DIR_PATH = fs::path(SOURCE_DIR) / "alist_sparse_matrices";
const fs::path RESULTS_DIR_PATH = fs::path(SOURCE_DIR) / "results";


config_data CFG;

struct sim_input
{
    size_t sim_number{};
    fs::path matrix_path{};
    std::vector<double> QBER{};
    H_matrix matrix{};
};

// Result of sum-product algorithm
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
    double actual_QBER{};                       // An accurate QBER that corresponds to the number of errors in the key.
    size_t max_iterations_successful_sp{};      // The maximum number of iterations of the sum-product algorithm in which Alice's syndrome matched Bob's syndrome.
    double ratio_trials_successful_sp{};        // Success rate of the sum-product algorithm. Success when Bob's syndrome matches Alice's.
    double ratio_trials_successful_ldpc{};      // Success rate of the QKD LDPC error reconciliation. Success when Bob and Alice's keys match.
};


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

// Records the results of the simulation in a ".csv" format file
void write_file(const std::vector<sim_result> &data, fs::path directory)
{
    try
    {
        std::string filename = "ldpc(trial_num=" + std::to_string(CFG.TRIALS_NUMBER) + ",max_sum_prod_iters=" +
                               std::to_string(CFG.SUM_PRODUCT_MAX_ITERATIONS) + ",seed=" + std::to_string(CFG.SIMULATION_SEED) + ").csv";
        fs::path result_file_path = directory / filename;

        std::fstream fout;
        fout.open(result_file_path, std::ios::out | std::ios::trunc);
        fout << "№;MATRIX_FILENAME;CODE_RATE;QBER;MAX_ITERATIONS_SUCCESSFUL_SUM_PRODUCT;RATIO_TRIALS_SUCCESSFUL_SUM_PRODUCT;RATIO_TRIALS_SUCCESSFUL_LDPC\n";
        for (size_t i = 0; i < data.size(); i++)
        {
            fout << data[i].sim_number << ";" << data[i].matrix_filename << ";" << data[i].code_rate << ";" << data[i].actual_QBER
                 << ";" << data[i].max_iterations_successful_sp << ";" << data[i].ratio_trials_successful_sp << ";" << data[i].ratio_trials_successful_ldpc << "\n";
        }
        fout.close();
    }
    catch (const std::exception &ex)
    {
        fmt::print(stderr, fg(fmt::color::red), "An error occurred while writing to the file.\n");
        throw;
    }
}

// Get QBER range based on code rate of matrix. R_QBER_parameters must be sorted. Looks for the first set of parameters
// where the code rate is less than or equal to the specified rate, and uses these parameters to generate a range of QBER values.
std::vector<double> get_rate_based_QBER_range(double code_rate, const std::vector<R_QBER_params> &R_QBER_parameters)
{
    std::vector<double> QBER;
    for (size_t i = 0; i < R_QBER_parameters.size(); i++)
    {
        if (code_rate <= R_QBER_parameters[i].code_rate || i == R_QBER_parameters.size() - 1)
        {
            for (double value = R_QBER_parameters[i].QBER_begin; value <= R_QBER_parameters[i].QBER_end;
                 value += R_QBER_parameters[i].QBER_step)
            {
                QBER.push_back(value);
            }
            break;
        }
    }
    if (QBER.empty())
    {
        throw std::runtime_error("An error occurred when generating a QBER range based on code rate.");
    }
    return QBER;
}


SP_result sum_product_decoding_regular(const double *const bit_array_llr, const H_matrix &matrix, const int *const syndrome,
                                       const size_t &max_num_iterations, const double &msg_threshold, int *const bit_array_out)
{
    double max_llr_c2b = 0.;
    double max_llr_b2c = 0.;
    double max_llr = 0.;

    double **bit_to_check_msg = new double *[matrix.num_check_nodes];
    for (size_t i = 0; i < matrix.num_check_nodes; i++)
    {
        bit_to_check_msg[i] = new double[matrix.max_check_nodes_weight];
        for (size_t j = 0; j < matrix.max_check_nodes_weight; j++)
        {
            bit_to_check_msg[i][j] = bit_array_llr[matrix.check_nodes[i][j]]; // Initialization
        }
    }

    double **check_to_bit_msg = new double *[matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        check_to_bit_msg[i] = new double[matrix.max_bit_nodes_weight];
    }

    double prod;
    double row_prod;
    int curr_bit_pos;

    int *check_pos_idx = new int[matrix.num_bit_nodes];
    double *total_bit_llr = new double[matrix.num_bit_nodes];
    int *decision_syndrome = new int[matrix.num_check_nodes];
    int *bit_pos_idx = new int[matrix.num_check_nodes];

    double sum;
    double col_sum;
    int curr_check_pos;

    size_t curr_iteration = 0;
    while (curr_iteration != max_num_iterations)
    {
        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\n\nIteration: {}\n", curr_iteration + 1);
        }

        // Compute extrinsic messages from check nodes to bit nodes (Step 1: Check messages)
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

        if (CFG.ENABLE_SUM_PRODUCT_MSG_LLR_THRESHOLD)
        {
            threshold_matrix_regular(check_to_bit_msg, matrix.num_bit_nodes, matrix.max_bit_nodes_weight, msg_threshold);
        }
        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\nE:\n");
            print_regular_matrix(check_to_bit_msg, matrix.num_bit_nodes, matrix.max_bit_nodes_weight);
        }

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

        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\nL:\n");
            print_array(total_bit_llr, matrix.num_bit_nodes);
            fmt::print(fg(fmt::color::blue), "\n\nz:\n");
            print_array(bit_array_out, matrix.num_bit_nodes);
        }

        calculate_syndrome_regular(bit_array_out, matrix, decision_syndrome);

        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\n\ns:\n");
            print_array(decision_syndrome, matrix.num_check_nodes);
        }

        if (arrays_equal(syndrome, decision_syndrome, matrix.num_check_nodes))
        {
            if (CFG.TRACE_SUM_PRODUCT_LLR)
            {
                fmt::print(fg(fmt::color::blue), "\n\nMAX_LLR = {}\n", max_llr);
            }
            free_matrix(bit_to_check_msg, matrix.num_check_nodes);
            free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
            delete[] bit_pos_idx;
            delete[] check_pos_idx;
            delete[] total_bit_llr;
            delete[] decision_syndrome;
            return {curr_iteration + 1, true};
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

        if (CFG.ENABLE_SUM_PRODUCT_MSG_LLR_THRESHOLD)
        {
            threshold_matrix_regular(bit_to_check_msg, matrix.num_check_nodes, matrix.max_check_nodes_weight, msg_threshold);
        }
        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\n\nM:\n");
            print_regular_matrix(bit_to_check_msg, matrix.num_check_nodes, matrix.max_check_nodes_weight);
        }
        if (CFG.TRACE_SUM_PRODUCT_LLR)
        {
            max_llr_c2b = get_max_llr_regular(check_to_bit_msg, matrix.max_bit_nodes_weight, matrix.num_bit_nodes);
            max_llr_b2c = get_max_llr_regular(bit_to_check_msg, matrix.max_check_nodes_weight, matrix.num_check_nodes);
            max_llr = std::max({max_llr, max_llr_c2b, max_llr_b2c});
        }

        curr_iteration++;
    }

    if (CFG.TRACE_SUM_PRODUCT_LLR)
    {
        fmt::print(fg(fmt::color::blue), "\n\nMAX_LLR = {}\n", max_llr);
    }

    free_matrix(bit_to_check_msg, matrix.num_check_nodes);
    free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
    delete[] bit_pos_idx;
    delete[] check_pos_idx;
    delete[] total_bit_llr;
    delete[] decision_syndrome;

    return {max_num_iterations, false};
}

SP_result sum_product_decoding_irregular(const double *const bit_array_llr, const H_matrix &matrix, const int *const syndrome,
                                         const size_t &max_num_iterations, const double &msg_threshold, int *const bit_array_out)
{
    double max_llr_c2b = 0.;
    double max_llr_b2c = 0.;
    double max_llr = 0.;

    double **bit_to_check_msg = new double *[matrix.num_check_nodes];
    for (size_t i = 0; i < matrix.num_check_nodes; i++)
    {
        bit_to_check_msg[i] = new double[matrix.check_nodes_weight[i]];
        for (size_t j = 0; j < matrix.check_nodes_weight[i]; j++)
        {
            bit_to_check_msg[i][j] = bit_array_llr[matrix.check_nodes[i][j]]; // Initialization
        }
    }

    double **check_to_bit_msg = new double *[matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        check_to_bit_msg[i] = new double[matrix.bit_nodes_weight[i]];
    }

    double prod;
    double row_prod;
    int curr_bit_pos;

    int *check_pos_idx = new int[matrix.num_bit_nodes];
    double *total_bit_llr = new double[matrix.num_bit_nodes];
    int *decision_syndrome = new int[matrix.num_check_nodes];
    int *bit_pos_idx = new int[matrix.num_check_nodes];

    double sum;
    double col_sum;
    int curr_check_pos;

    size_t curr_iteration = 0;
    while (curr_iteration != max_num_iterations)
    {
        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\n\nIteration: {}\n", curr_iteration + 1);
        }

        // Compute extrinsic messages from check nodes to bit nodes (Step 1: Check messages)
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

        if (CFG.ENABLE_SUM_PRODUCT_MSG_LLR_THRESHOLD)
        {
            threshold_matrix_irregular(check_to_bit_msg, matrix.num_bit_nodes, matrix.bit_nodes_weight, msg_threshold);
        }
        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\nE:\n");
            print_irregular_matrix(check_to_bit_msg, matrix.num_bit_nodes, matrix.bit_nodes_weight);
        }

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

        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\nL:\n");
            print_array(total_bit_llr, matrix.num_bit_nodes);
            fmt::print(fg(fmt::color::blue), "\n\nz:\n");
            print_array(bit_array_out, matrix.num_bit_nodes);
        }

        calculate_syndrome_irregular(bit_array_out, matrix, decision_syndrome);

        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\n\ns:\n");
            print_array(decision_syndrome, matrix.num_check_nodes);
        }

        if (arrays_equal(syndrome, decision_syndrome, matrix.num_check_nodes))
        {
            if (CFG.TRACE_SUM_PRODUCT_LLR)
            {
                fmt::print(fg(fmt::color::blue), "\n\nMAX_LLR = {}\n", max_llr);
            }
            free_matrix(bit_to_check_msg, matrix.num_check_nodes);
            free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
            delete[] bit_pos_idx;
            delete[] check_pos_idx;
            delete[] total_bit_llr;
            delete[] decision_syndrome;
            return {curr_iteration + 1, true};
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

        if (CFG.ENABLE_SUM_PRODUCT_MSG_LLR_THRESHOLD)
        {
            threshold_matrix_irregular(bit_to_check_msg, matrix.num_check_nodes, matrix.check_nodes_weight, msg_threshold);
        }
        if (CFG.TRACE_SUM_PRODUCT)
        {
            fmt::print(fg(fmt::color::blue), "\n\nM:\n");
            print_irregular_matrix(bit_to_check_msg, matrix.num_check_nodes, matrix.check_nodes_weight);
        }
        if (CFG.TRACE_SUM_PRODUCT_LLR)
        {
            max_llr_c2b = get_max_llr_irregular(check_to_bit_msg, matrix.bit_nodes_weight, matrix.num_bit_nodes);
            max_llr_b2c = get_max_llr_irregular(bit_to_check_msg, matrix.check_nodes_weight, matrix.num_check_nodes);
            max_llr = std::max({max_llr, max_llr_c2b, max_llr_b2c});
        }

        curr_iteration++;
    }

    if (CFG.TRACE_SUM_PRODUCT_LLR)
    {
        fmt::print(fg(fmt::color::blue), "\n\nMAX_LLR = {}\n", max_llr);
    }

    free_matrix(bit_to_check_msg, matrix.num_check_nodes);
    free_matrix(check_to_bit_msg, matrix.num_bit_nodes);
    delete[] bit_pos_idx;
    delete[] check_pos_idx;
    delete[] total_bit_llr;
    delete[] decision_syndrome;

    return {max_num_iterations, false};
}

LDPC_result QKD_LDPC_regular(const int *const alice_bit_array, const int *const bob_bit_array, const double &QBER, const H_matrix &matrix)
{
    double log_p = log((1. - QBER) / QBER);
    double *apriori_llr = new double[matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        apriori_llr[i] = (bob_bit_array[i] ? -log_p : log_p);
    }

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\nr:\n");
        print_array(apriori_llr, matrix.num_bit_nodes);
    }

    int *alice_syndrome = new int[matrix.num_check_nodes];
    calculate_syndrome_regular(alice_bit_array, matrix, alice_syndrome);

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\n\nAlice syndrome:\n");
        print_array(alice_syndrome, matrix.num_check_nodes);
    }

    int *bob_solution = new int[matrix.num_bit_nodes];
    LDPC_result ldpc_res;
    ldpc_res.sp_res = sum_product_decoding_regular(apriori_llr, matrix, alice_syndrome, CFG.SUM_PRODUCT_MAX_ITERATIONS,
                                                   CFG.SUM_PRODUCT_MSG_LLR_THRESHOLD, bob_solution);

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\nBob corrected bit array:\n");
        print_array(bob_solution, matrix.num_bit_nodes);
    }

    ldpc_res.keys_match = arrays_equal(alice_bit_array, bob_solution, matrix.num_bit_nodes);

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\n\nIterations performed: {}\n", ldpc_res.sp_res.iterations_num);
        fmt::print(fg(fmt::color::blue), "Syndromes are match: {}\n", ((ldpc_res.sp_res.syndromes_match) ? "YES" : "NO"));
        fmt::print(fg(fmt::color::blue), "Keys are match: {}\n", ((ldpc_res.keys_match) ? "YES" : "NO"));
    }

    delete[] apriori_llr;
    delete[] alice_syndrome;
    delete[] bob_solution;

    return ldpc_res;
}

LDPC_result QKD_LDPC_irregular(const int *const alice_bit_array, const int *const bob_bit_array, const double &QBER, const H_matrix &matrix)
{
    double log_p = log((1. - QBER) / QBER);
    double *apriori_llr = new double[matrix.num_bit_nodes];
    for (size_t i = 0; i < matrix.num_bit_nodes; i++)
    {
        apriori_llr[i] = (bob_bit_array[i] ? -log_p : log_p);
    }

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\nr:\n");
        print_array(apriori_llr, matrix.num_bit_nodes);
    }

    int *alice_syndrome = new int[matrix.num_check_nodes];
    calculate_syndrome_irregular(alice_bit_array, matrix, alice_syndrome);

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\n\nAlice syndrome:\n");
        print_array(alice_syndrome, matrix.num_check_nodes);
    }

    int *bob_solution = new int[matrix.num_bit_nodes];
    LDPC_result ldpc_res;
    ldpc_res.sp_res = sum_product_decoding_irregular(apriori_llr, matrix, alice_syndrome, CFG.SUM_PRODUCT_MAX_ITERATIONS,
                                                     CFG.SUM_PRODUCT_MSG_LLR_THRESHOLD, bob_solution);

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\nBob corrected bit array:\n");
        print_array(bob_solution, matrix.num_bit_nodes);
    }

    ldpc_res.keys_match = arrays_equal(alice_bit_array, bob_solution, matrix.num_bit_nodes);

    if (CFG.TRACE_QKD_LDPC)
    {
        fmt::print(fg(fmt::color::blue), "\n\nIterations performed: {}\n", ldpc_res.sp_res.iterations_num);
        fmt::print(fg(fmt::color::blue), "Syndromes are match: {}\n", ((ldpc_res.sp_res.syndromes_match) ? "YES" : "NO"));
        fmt::print(fg(fmt::color::blue), "Keys are match: {}\n", ((ldpc_res.keys_match) ? "YES" : "NO"));
    }

    delete[] apriori_llr;
    delete[] alice_syndrome;
    delete[] bob_solution;

    return ldpc_res;
}

// Interactive simulation of quantum key distribution (QKD) using LDPC codes.
void QKD_LDPC_interactive_simulation(fs::path matrix_dir_path)
{
    H_matrix matrix;
    std::vector<fs::path> matrix_paths = get_file_paths_in_directory(matrix_dir_path);
    fs::path matrix_path = select_matrix_file(matrix_paths);

    if (CFG.USE_DENSE_MATRICES)
    {
        read_dense_matrix(matrix_path, matrix);
    }
    else
    {
        read_sparse_alist_matrix(matrix_path, matrix);
    }

    fmt::print(fg(fmt::color::green), "{}\n", ((matrix.is_regular) ? "Matrix H is regular." : "Matrix H is irregular."));

    size_t num_check_nodes = matrix.num_check_nodes;
    size_t num_bit_nodes = matrix.num_bit_nodes;
    int *alice_bit_array = new int[num_bit_nodes];
    int *bob_bit_array = new int[num_bit_nodes];

    std::mt19937 prng(CFG.SIMULATION_SEED); 
    double code_rate = static_cast<double>(matrix.num_check_nodes) / matrix.num_bit_nodes;
    std::vector<double> QBER = get_rate_based_QBER_range(code_rate, CFG.R_QBER_PARAMETERS);
    for (size_t i = 0; i < QBER.size(); i++)
    {
        fmt::print(fg(fmt::color::green), "№:{}\n", i + 1);

        generate_random_bit_array(prng, num_bit_nodes, alice_bit_array);
        double actual_QBER = introduce_errors(prng, alice_bit_array, num_bit_nodes, QBER[i], bob_bit_array);
        fmt::print(fg(fmt::color::green), "Actual QBER: {}\n", actual_QBER);

        if (actual_QBER == 0.)
        {
            free_matrix_H(matrix);
            delete[] alice_bit_array;
            delete[] bob_bit_array;
            throw std::runtime_error("Key size '" + std::to_string(num_bit_nodes) + "' is too small for QBER.");
        }

        int error_num = 0;
        for (size_t i = 0; i < num_bit_nodes; i++)
        {
            error_num += alice_bit_array[i] ^ bob_bit_array[i];
        }
        fmt::print(fg(fmt::color::green), "Number of errors in a key: {}\n", error_num);

        LDPC_result try_result;
        if (matrix.is_regular)
        {
            try_result = QKD_LDPC_regular(alice_bit_array, bob_bit_array, actual_QBER, matrix);
        }
        else
        {
            try_result = QKD_LDPC_irregular(alice_bit_array, bob_bit_array, actual_QBER, matrix);
        }
        fmt::print(fg(fmt::color::green), "Iterations performed: {}\n", try_result.sp_res.iterations_num);
        fmt::print(fg(fmt::color::green), "{}\n\n", ((try_result.keys_match && try_result.sp_res.syndromes_match) ? "Error reconciliation SUCCESSFUL" : "Error reconciliation FAILED"));
    }

    free_matrix_H(matrix);
    delete[] alice_bit_array;
    delete[] bob_bit_array;
}

// Prepares input data for batch simulation.
void prepare_sim_inputs(const std::vector<fs::path> &matrix_paths, std::vector<sim_input> &sim_inputs_out)
{
    for (size_t i = 0; i < matrix_paths.size(); i++)
    {
        if (CFG.USE_DENSE_MATRICES)
        {
            read_dense_matrix(matrix_paths[i], sim_inputs_out[i].matrix);
        }
        else
        {
            read_sparse_alist_matrix(matrix_paths[i], sim_inputs_out[i].matrix);
        }

        sim_inputs_out[i].sim_number = i;
        sim_inputs_out[i].matrix_path = matrix_paths[i];

        double code_rate = static_cast<double>(sim_inputs_out[i].matrix.num_check_nodes) / sim_inputs_out[i].matrix.num_bit_nodes;
        sim_inputs_out[i].QBER = get_rate_based_QBER_range(code_rate, CFG.R_QBER_PARAMETERS);
    }
}

// Runs a single QKD LDPC trial.
trial_result run_trial(const H_matrix &matrix, double QBER, size_t seed)
{
    std::mt19937 prng(seed);

    trial_result result;
    int *alice_bit_array = new int[matrix.num_bit_nodes];
    int *bob_bit_array = new int[matrix.num_bit_nodes];
    generate_random_bit_array(prng, matrix.num_bit_nodes, alice_bit_array);
    result.actual_QBER = introduce_errors(prng, alice_bit_array, matrix.num_bit_nodes, QBER, bob_bit_array);
    if (result.actual_QBER == 0.)
    {
        delete[] alice_bit_array;
        delete[] bob_bit_array;
        throw std::runtime_error("Key size '" + std::to_string(matrix.num_bit_nodes) + "' is too small for QBER.");
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

// Distributes all combinations of the experiment evenly across the CPU threads and runs it.
std::vector<sim_result> QKD_LDPC_batch_simulation(const std::vector<sim_input> &sim_in)
{
    using namespace indicators;
    size_t sim_total = 0;
    for (size_t i = 0; i < sim_in.size(); i++)
    {
        sim_total += sim_in[i].QBER.size();     // For each matrix, keys are generated with error rates given in the QBER vector
    }

    size_t trials_total = sim_total * CFG.TRIALS_NUMBER;    
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
        option::MaxProgress{trials_total}};

    size_t curr_sim = 0;
    size_t iteration = 0;
    std::vector<sim_result> sim_results(sim_total);
    std::vector<trial_result> trial_results(CFG.TRIALS_NUMBER);

    std::mt19937 prng(CFG.SIMULATION_SEED);
    std::uniform_int_distribution<size_t> distribution(0, std::numeric_limits<size_t>::max());

    BS::thread_pool pool(CFG.THREADS_NUMBER);
    for (size_t i = 0; i < sim_in.size(); i++)
    {
        const H_matrix &matrix = sim_in[i].matrix;
        double code_rate = static_cast<double>(matrix.num_check_nodes) / matrix.num_bit_nodes;
        std::string matrix_filename = sim_in[i].matrix_path.filename().string();
        for (size_t j = 0; j < sim_in[i].QBER.size(); j++)
        {
            iteration += CFG.TRIALS_NUMBER;
            bar.set_option(option::PostfixText{
                std::to_string(iteration) + "/" + std::to_string(trials_total)});

            double QBER = sim_in[i].QBER[j];
            // TRIALS_NUMBER of trials are performed with each combination to calculate the mean values
            pool.detach_loop<size_t>(0, CFG.TRIALS_NUMBER,
                                     [&matrix, &QBER, &trial_results, &prng, &distribution, &bar](size_t k)
                                     {
                                         trial_results[k] = run_trial(matrix, QBER, distribution(prng));
                                         bar.tick(); // For correct time estimation
                                     });
            pool.wait();

            size_t trials_successful_sp = 0;
            size_t trials_successful_ldpc = 0;
            size_t max_iterations_successful_sp = 0;
            size_t curr_sp_iterations_num{};
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
            sim_results[curr_sim].ratio_trials_successful_ldpc = static_cast<double>(trials_successful_ldpc) / CFG.TRIALS_NUMBER;
            sim_results[curr_sim].ratio_trials_successful_sp = static_cast<double>(trials_successful_sp) / CFG.TRIALS_NUMBER;
            curr_sim++;
        }
    }
    return sim_results;
}

int main()
{
    std::vector<sim_input> sim_inputs;
    std::vector<sim_result> sim_results;

    try
    {
        CFG = get_config_data(CONFIG_PATH);
        fs::path matrix_dir_path = ((CFG.USE_DENSE_MATRICES) ? DENSE_MATRIX_DIR_PATH : ALIST_MATRIX_DIR_PATH);
        if (CFG.INTERACTIVE_MODE)
        {
            fmt::print(fg(fmt::color::purple), "INTERACTIVE MODE\n");
            QKD_LDPC_interactive_simulation(matrix_dir_path);
        }
        else
        {
            fmt::print(fg(fmt::color::purple), "BATCH MODE\n");
            std::vector<fs::path> matrix_paths = get_file_paths_in_directory(matrix_dir_path);
            if (matrix_paths.empty())
            {
                throw std::runtime_error("Matrix folder is empty: " + matrix_dir_path.string());
            }

            sim_inputs.resize(matrix_paths.size());
            prepare_sim_inputs(matrix_paths, sim_inputs);

            sim_results = QKD_LDPC_batch_simulation(sim_inputs);

            for (size_t i = 0; i < sim_inputs.size(); i++)
            {
                free_matrix_H(sim_inputs[i].matrix);
            }
            sim_inputs.clear();

            fmt::print(fg(fmt::color::green), "Dynamic memory for storing matrices has been successfully freed!\n");
            fmt::print(fg(fmt::color::green), "The results will be written to the directory: {}\n", RESULTS_DIR_PATH.string());
            write_file(sim_results, RESULTS_DIR_PATH);
        }
    }
    catch (const std::exception &e)
    {
        for (size_t i = 0; i < sim_inputs.size(); i++)
        {
            free_matrix_H(sim_inputs[i].matrix);
        }
        sim_inputs.clear();
        fmt::print(fg(fmt::color::green), "Dynamic memory for storing matrices has been successfully freed!\n");

        fmt::print(stderr, fg(fmt::color::red), "ERROR: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
