#pragma once
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <XoshiroCpp.hpp>
#include <BS_thread_pool.hpp>
#include <indicators/progress_bar.hpp>
#include <indicators/cursor_control.hpp>

#include "config.hpp"
#include "qkd_ldpc_algorithm.hpp"

struct sim_input
{
    fs::path matrix_path{};
    std::vector<double> QBER{};
    H_matrix matrix{};
};

struct trial_result
{
    LDPC_result ldpc_res{};
    double initial_QBER{};
};

struct sim_result
{
    size_t sim_number{};
    std::string matrix_filename{};
    bool is_regular{};                          // Matrix type.
    size_t num_bit_nodes{};                     // Number of bit nodes, which is defined as the number of columns in the parity check matrix.
    size_t num_check_nodes{};                   // Number of check nodes, which is defined as the number of rows in the parity check matrix.
    double initial_QBER{};                       // An accurate QBER that corresponds to the number of errors in the key.
    size_t iterations_successful_sp_max{};      // The maximum number of iterations of the sum-product algorithm in which Alice's syndrome matched Bob's syndrome (i.e. successful).
    size_t iterations_successful_sp_min{};      // The minimum number of iterations of the sum-product algorithm.
    double iterations_successful_sp_mean{};     // The mean number of iterations of the sum-product algorithm. 
    double iterations_successful_sp_std_dev{};  // The standard deviation of iterations of the sum-product algorithm. 
    double ratio_trials_successful_sp{};        // Success rate of the sum-product algorithm. Success when Bob's syndrome matches Alice's.
    double ratio_trials_successful_ldpc{};      // Success rate of the QKD LDPC error reconciliation. Success when Bob and Alice's keys match.
};

void write_file(const std::vector<sim_result> &data, fs::path directory);
std::vector<double> get_rate_based_QBER_range(const double code_rate, const std::vector<R_QBER_params> &R_QBER_parameters);
void QKD_LDPC_interactive_simulation(fs::path matrix_dir_path);
void prepare_sim_inputs(const std::vector<fs::path> &matrix_paths, std::vector<sim_input> &sim_inputs_out);
trial_result run_trial(const H_matrix &matrix, const double QBER, size_t seed);
std::vector<sim_result> QKD_LDPC_batch_simulation(const std::vector<sim_input> &sim_in);
