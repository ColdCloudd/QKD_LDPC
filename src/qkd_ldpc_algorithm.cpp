#include "qkd_ldpc_algorithm.hpp"

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
