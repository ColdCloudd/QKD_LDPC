#include <filesystem>

#include "config.hpp"
#include "simulation.hpp"

namespace fs = std::filesystem;

const fs::path CONFIG_PATH = fs::path(SOURCE_DIR) / "config.json";
const fs::path DENSE_MATRIX_DIR_PATH = fs::path(SOURCE_DIR) / "dense_matrices";
const fs::path ALIST_MATRIX_DIR_PATH = fs::path(SOURCE_DIR) / "alist_sparse_matrices";
const fs::path RESULTS_DIR_PATH = fs::path(SOURCE_DIR) / "results";

config_data CFG;

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
