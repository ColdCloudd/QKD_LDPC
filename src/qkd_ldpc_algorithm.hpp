#pragma once

#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/ranges.h>

#include "config.hpp"
#include "utils.hpp"
#include "array_and_matrix_operations.hpp"

namespace fs = std::filesystem;

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

SP_result sum_product_decoding_regular(const double *const bit_array_llr, const H_matrix &matrix, const int *const syndrome,
                                       const size_t &max_num_iterations, const double &msg_threshold, int *const bit_array_out);
SP_result sum_product_decoding_irregular(const double *const bit_array_llr, const H_matrix &matrix, const int *const syndrome,
                                         const size_t &max_num_iterations, const double &msg_threshold, int *const bit_array_out);
LDPC_result QKD_LDPC_regular(const int *const alice_bit_array, const int *const bob_bit_array, const double &QBER, const H_matrix &matrix);
LDPC_result QKD_LDPC_irregular(const int *const alice_bit_array, const int *const bob_bit_array, const double &QBER, const H_matrix &matrix);