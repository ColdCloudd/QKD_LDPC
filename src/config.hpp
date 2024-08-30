#pragma once
#include <vector>
#include <fstream>
#include <filesystem>

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Structure that stores code rate values that correspond to a range of QBER values from QBER_begin to QBER_end in QBER_step increments.
struct R_QBER_params
{
    double code_rate{};
    double QBER_begin{};
    double QBER_end{};
    double QBER_step{};
};

struct config_data
{
    // Number of threads for parallelizing runs
    size_t THREADS_NUMBER{};

    // Number of runs with one combination
    size_t TRIALS_NUMBER{};

    // Seed of simulation
    size_t SIMULATION_SEED{};

    // If true, the user can select the parity check matrix and run tests with it. Otherwise, the BATCH MODE works, 
    // which reads all parity check matrices from the specified directory and runs tests with them and writes the results to a file.
    bool INTERACTIVE_MODE{};

    // The maximum number of iterations of the sum-product algorithm. 
    // If the maximum number of iterations is reached, error reconciliation in the key is considered unsuccessful.
    size_t SUM_PRODUCT_MAX_ITERATIONS{};

    // Use dense matrices (folder dense_matrices) instead of sparse matrices (folder alist_sparse_matrices).
    bool USE_DENSE_MATRICES{};

    // Output intermediate results of LDPC operation to the console.
    bool TRACE_QKD_LDPC{};

    // Output intermediate results of the sum-product algorithm to the console.
    bool TRACE_SUM_PRODUCT{};

    // Console output of maximum log likelihood ratios (LLR) values of the message during the sum-product algorithm.
    bool TRACE_SUM_PRODUCT_LLR{};

    // Enables limitation on the maximum LLR value of the message, which cannot exceed the set threshold. 
    // If the set threshold is exceeded, the value is set equal to SUM_PRODUCT_MSG_LLR_THRESHOLD.
    bool ENABLE_SUM_PRODUCT_MSG_LLR_THRESHOLD{};

    // The maximum LLR value a message can have.
    double SUM_PRODUCT_MSG_LLR_THRESHOLD{};

    // Code rate and QBER correspondence set.
    std::vector<R_QBER_params> R_QBER_PARAMETERS{};
};

extern config_data CFG;

config_data get_config_data(fs::path config_path);