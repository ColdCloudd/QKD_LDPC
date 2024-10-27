#include "array_and_matrix_operations.hpp"

// Convert a dense parity check matrix into an array containing information about bit nodes and associated check nodes (sparse matrix).
void get_bit_nodes(const std::vector<std::vector<int>> &matrix, const int *const bit_nodes_weight, int **&bit_nodes_out)
{
    size_t num_bit_nodes = matrix[0].size();
    size_t num_check_nodes = matrix.size();

    size_t n;
    bit_nodes_out = new int *[num_bit_nodes];
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

// Convert a dense parity check matrix into an array containing information about check nodes and associated bit nodes (sparse matrix).
void get_check_nodes(const std::vector<std::vector<int>> &matrix, const int *const check_nodes_weight, int **&check_nodes_out)
{
    size_t num_bit_nodes = matrix[0].size();
    size_t num_check_nodes = matrix.size();

    size_t n;
    check_nodes_out = new int *[num_check_nodes];
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

// Finding the maximum modulo LLR value in a given regular matrix.
double get_max_llr_regular(const double *const *matrix, const size_t &nodes_weight, const size_t &rows_number)
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

// Finding the maximum modulo LLR value in a given irregular matrix.
double get_max_llr_irregular(const double *const *matrix, const int *const nodes_weight, const size_t &rows_number)
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

// Freeing memory allocated for the parity check matrix.
void free_matrix_H(H_matrix &matrix)
{
    free_matrix(matrix.bit_nodes, matrix.num_bit_nodes);
    free_matrix(matrix.check_nodes, matrix.num_check_nodes);
    delete[] matrix.bit_nodes_weight;
    delete[] matrix.check_nodes_weight;
}

bool arrays_equal(const int *const array1, const int *const array2, const size_t &array_length)
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

// Function for reading a sparse matrix from a file in alist format (https://rptu.de/channel-codes/matrix-file-formats).
void read_sparse_alist_matrix(const fs::path &matrix_path, H_matrix &matrix_out)
{
    std::vector<std::string> line_vec;
    std::ifstream file(matrix_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + matrix_path.string());
    }
    std::string line;

    while (getline(file, line))
    {
        line_vec.push_back(line);
    }
    file.close();

    if (line_vec.empty())
    {
        throw std::runtime_error("File is empty or cannot be read properly: " + matrix_path.string());
    }

    std::vector<std::vector<int>> vec_int;
    try
    {
        for (const auto &line : line_vec)
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
    catch (const std::exception &e)
    {
        fmt::print(stderr, fg(fmt::color::red), "An error occurred while parsing file: {}\n", matrix_path.string());
        throw;
    }

    if (vec_int.size() < 4)
    {
        throw std::runtime_error("Insufficient data in the file: " + matrix_path.string());
    }

    if (vec_int[0].size() != 2 || vec_int[1].size() != 2)
    {
        throw std::runtime_error("File format does not match the alist format: " + matrix_path.string());
    }

    size_t col_num = vec_int[0][0];     // n
    size_t row_num = vec_int[0][1];     // m

    size_t max_col_weight = vec_int[1][0];      // d^(v)_max
    size_t max_row_weight = vec_int[1][1];      // d^(c)_max

    size_t num_bit_nodes = vec_int[2].size();
    size_t num_check_nodes = vec_int[3].size();

    size_t curr_line = 4;

    if (vec_int.size() < curr_line + num_bit_nodes + num_check_nodes)
    {
        throw std::runtime_error("Insufficient data in the file: " + matrix_path.string());
    }

    if (col_num != num_bit_nodes)
    {
        throw std::runtime_error("Number of columns '" + std::to_string(col_num) + "' is not the same as the length of the third line '" + std::to_string(num_bit_nodes) + "'. File: " + matrix_path.string());
    }
    else if (row_num != num_check_nodes)
    {
        throw std::runtime_error("Number of rows '" + std::to_string(row_num) + "' is not the same as the length of the fourth line '" + std::to_string(num_check_nodes) + "'. File: " + matrix_path.string());
    }

    // Initialization of node weights and check for matrix regularity
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

    // Check if the number of non-zero elements corresponds to the node weights
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
            free_matrix_H(matrix_out);
            throw std::runtime_error("Number of non-zero elements '" + std::to_string(non_zero_num) + "' in the line '" + std::to_string(curr_line + i + 1) + "' does not match the weight in the third line '" + std::to_string(vec_int[2][i]) + "'. File: " + matrix_path.string());
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
            free_matrix_H(matrix_out);
            throw std::runtime_error("Number of non-zero elements '" + std::to_string(non_zero_num) + "' in the line '" + std::to_string(curr_line + i + 1) + "' does not match the weight in the fourth line '" + std::to_string(vec_int[3][i]) + "'. File: " + matrix_path.string());
        }
    }

    // Filling the matrix of bit nodes
    try
    {
        curr_line = 4;
        matrix_out.bit_nodes = new int *[num_bit_nodes];
        for (size_t i = 0; i < num_bit_nodes; ++i)
        {
            matrix_out.bit_nodes[i] = new int[matrix_out.bit_nodes_weight[i]];
            for (size_t j = 0; j < matrix_out.bit_nodes_weight[i]; ++j)
            {
                matrix_out.bit_nodes[i][j] = (vec_int[curr_line + i][j] - 1);
            }
        }
    }
    catch (const std::exception &e)
    {
        free_matrix_H(matrix_out);
        fmt::print(stderr, fg(fmt::color::red), "An error occurred while creating 'bit_nodes' matrix from file: {}\n", matrix_path.string());
        throw;
    }

    // Filling the matrix of check nodes
    try
    {
        curr_line += num_bit_nodes;
        matrix_out.check_nodes = new int *[num_check_nodes];
        for (size_t i = 0; i < num_check_nodes; ++i)
        {
            matrix_out.check_nodes[i] = new int[matrix_out.check_nodes_weight[i]];
            for (size_t j = 0; j < matrix_out.check_nodes_weight[i]; ++j)
            {
                matrix_out.check_nodes[i][j] = (vec_int[curr_line + i][j] - 1);
            }
        }
    }
    catch (const std::exception &e)
    {
        free_matrix_H(matrix_out);
        fmt::print(stderr, fg(fmt::color::red), "An error occurred while creating 'check_nodes' matrix from file: {}\n", matrix_path.string());
        throw;
    }

    matrix_out.num_check_nodes = row_num;
    matrix_out.num_bit_nodes = col_num;
    matrix_out.max_check_nodes_weight = max_row_weight;
    matrix_out.max_bit_nodes_weight = max_col_weight;
    matrix_out.is_regular = is_regular;
}

// Read dense matrix from file.
void read_dense_matrix(const fs::path &matrix_path, H_matrix &matrix_out)
{
    std::vector<std::string> line_vec;
    std::ifstream file(matrix_path);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + matrix_path.string());
    }
    std::string line;

    while (getline(file, line))
    {
        line_vec.push_back(line);
    }
    file.close();

    if (line_vec.empty())
    {
        throw std::runtime_error("File is empty or cannot be read properly: " + matrix_path.string());
    }

    std::vector<std::vector<int>> vec_int;
    try
    {
        for (const auto &line : line_vec)
        {
            std::istringstream iss(line);
            std::vector<int> numbers;
            int number;
            while (iss >> number)
            {
                if (number != 0 && number != 1)
                {
                    throw std::runtime_error("Parity check matrix can only take values ​​0 or 1.");
                }
                numbers.push_back(number);
            }
            vec_int.push_back(numbers);
        }
    }
    catch (const std::exception &e)
    {
        fmt::print(stderr, fg(fmt::color::red), "An error occurred while parsing file: {}\n", matrix_path.string());
        throw;
    }

    for (size_t i = 0; i < vec_int.size(); i++)
    {
        if (vec_int[0].size() != vec_int[i].size())
        {
            throw std::runtime_error("Different lengths of rows in a matrix. File: " + matrix_path.string());
        }
    }

    size_t col_num = vec_int[0].size();
    size_t row_num = vec_int.size();

    matrix_out.bit_nodes_weight = new int[col_num];
    matrix_out.check_nodes_weight = new int[row_num];

    // Counting column weights and determining the maximum weight
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
            free_matrix_H(matrix_out);
            throw std::runtime_error("Column '" + std::to_string(i + 1) + "' weight cannot be equal to or less than zero. File: " + matrix_path.string());
        }
        matrix_out.bit_nodes_weight[i] = curr_weight;
        if (curr_weight > max_col_weight)
        {
            max_col_weight = curr_weight;
        }
    }

    // Counting row weights and determining the maximum weight
    size_t max_row_weight = 0;
    for (size_t i = 0; i < row_num; i++)
    {
        curr_weight = accumulate(vec_int[i].begin(), vec_int[i].end(), 0);
        if (curr_weight <= 0)
        {
            free_matrix_H(matrix_out);
            throw std::runtime_error("Row '" + std::to_string(i + 1) + "' weight cannot be equal to or less than zero. File: " + matrix_path.string());
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

    // Filling bit and check nodes 
    get_bit_nodes(vec_int, matrix_out.bit_nodes_weight, matrix_out.bit_nodes);
    get_check_nodes(vec_int, matrix_out.check_nodes_weight, matrix_out.check_nodes);

    matrix_out.num_check_nodes = row_num;
    matrix_out.num_bit_nodes = col_num;
    matrix_out.max_check_nodes_weight = max_row_weight;
    matrix_out.max_bit_nodes_weight = max_col_weight;
    matrix_out.is_regular = is_regular;
}

// Generates Alice's key.
void generate_random_bit_array(XoshiroCpp::Xoshiro256PlusPlus &prng, size_t length, int *const random_bit_array_out)
{
    std::uniform_int_distribution<int> distribution(0, 1);
    for (int i = 0; i < length; ++i)
    {
        random_bit_array_out[i] = distribution(prng);
    }
}

// Generates Bob's key by making errors in Alice's key. Generates the exact number of errors in the key and returns the exact QBER.
double introduce_errors(XoshiroCpp::Xoshiro256PlusPlus &prng, const int *const bit_array, size_t array_length, double error_probability, int *const bit_array_with_errors_out)
{
    size_t num_errors = static_cast<size_t>(array_length * error_probability);
    if (num_errors == 0)
    {
        std::copy(bit_array, bit_array + array_length, bit_array_with_errors_out);
    }
    else
    {
        size_t *error_positions = new size_t[array_length];
        for (size_t i = 0; i < array_length; ++i)
        {
            error_positions[i] = i;
        }

        std::shuffle(error_positions, error_positions + array_length, prng);
        std::copy(bit_array, bit_array + array_length, bit_array_with_errors_out);

        for (size_t i = 0; i < num_errors; ++i)
        {
            bit_array_with_errors_out[error_positions[i]] ^= 1;
        }

        delete[] error_positions;
    }
    return static_cast<double>(num_errors) / array_length;
}

// Computes the key syndrome using a regular parity check matrix.
void calculate_syndrome_regular(const int *const bit_array, const H_matrix &matrix, int *const syndrome_out)
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

// Computes the key syndrome using a irregular parity check matrix.
void calculate_syndrome_irregular(const int *const bit_array, const H_matrix &matrix, int *const syndrome_out)
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

// Limiting the LLR values of messages in a regular matrix to a given threshold.
void threshold_matrix_regular(double *const *matrix, const size_t &rows_number, const size_t &nodes_weight, const double &msg_threshold)
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

// Limiting the LLR values of messages in irregular matrix to a given threshold.
void threshold_matrix_irregular(double *const *matrix, const size_t &rows_number, const int *const nodes_weight, const double &msg_threshold)
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
