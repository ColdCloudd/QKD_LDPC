#include "simulation.hpp"

// Records the results of the simulation in a ".csv" format file
void write_file(const std::vector<sim_result> &data, fs::path directory)
{
    try
    {
        if (!fs::exists(directory))
        {
            fs::create_directories(directory);
        }
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

        double code_rate = 1. - (static_cast<double>(sim_inputs_out[i].matrix.num_check_nodes) / sim_inputs_out[i].matrix.num_bit_nodes);
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
        double code_rate = 1. - (static_cast<double>(matrix.num_check_nodes) / matrix.num_bit_nodes);
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
