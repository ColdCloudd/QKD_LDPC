#include <filesystem>

#include "config.hpp"
#include "qkd_ldpc_algorithm.hpp"

namespace fs = std::filesystem;

const fs::path EXAMPLE_DENSE_MATRIX_PATH = fs::path(SOURCE_DIR) / "dense_matrices/(N=6,K=2,M=4,R=0.66).txt";

config_data CFG;

int main()
{
    H_matrix matrix;
    int *alice_bit_array;
    int *bob_bit_array;

    try
    {
        CFG.SUM_PRODUCT_MAX_ITERATIONS = 100;

        CFG.ENABLE_SUM_PRODUCT_MSG_LLR_THRESHOLD = true;
        CFG.SUM_PRODUCT_MSG_LLR_THRESHOLD = 100.;

        CFG.TRACE_QKD_LDPC = true;
        CFG.TRACE_SUM_PRODUCT = true;
        CFG.TRACE_SUM_PRODUCT_LLR = true;
        
        read_dense_matrix(EXAMPLE_DENSE_MATRIX_PATH, matrix);

        size_t num_check_nodes = matrix.num_check_nodes;
        size_t num_bit_nodes = matrix.num_bit_nodes;

        // Page 33, example 2.5: https://www.researchgate.net/publication/228977165_Introducing_Low-Density_Parity-Check_Codes
        alice_bit_array = new int[num_bit_nodes] {0, 0, 1, 0, 1, 1};
        bob_bit_array = new int[num_bit_nodes] {1, 0, 1, 0, 1, 1};

        double QBER = 0.2;
        QKD_LDPC_regular(alice_bit_array, bob_bit_array, QBER, matrix);
        
        free_matrix_H(matrix);
        delete[] alice_bit_array;
        delete[] bob_bit_array;
    }
    catch (const std::exception &e)
    {
        free_matrix_H(matrix);
        delete[] alice_bit_array;
        delete[] bob_bit_array;
        fmt::print(fg(fmt::color::green), "Dynamic memory for storing matrix has been successfully freed!\n");

        fmt::print(stderr, fg(fmt::color::red), "ERROR: {}\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}