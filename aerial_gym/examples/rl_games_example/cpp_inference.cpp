

// include file readers and IO modules

#include <iostream>
#include <fstream>
#include <string>

#include <eigen3/Eigen/Dense>

// include vector
#include <vector>
#include <map>


#include <chrono>

std::string weights_file_prefix = "networks/weight_";
std::string weights_file_suffix = ".txt";

std::string bias_file_prefix = "networks/bias_";
std::string bias_file_suffix = ".txt";

int num_layers = 3;


class NN_CPP
{
    public:
    std::vector<Eigen::MatrixXf> layer_weights;
    std::vector<Eigen::MatrixXf> layer_bias;


    NN_CPP(std::string weights_file_prefix, std::string weights_file_suffix, std::string bias_file_prefix, std::string bias_file_suffix, int num_layers)
    {
        std::string weights_filename;
        std::string bias_filename;
        
        for(int i = 0; i<= num_layers; ++i)
        {
            std::cout << "Now working with file: " << i << std::endl;
            std::vector<float> weights;
            std::vector<float> bias;
            int dim0, dim1;
            weights_filename = weights_file_prefix + std::to_string(i) + weights_file_suffix;
            std::ifstream weights_file(weights_filename);
            if(weights_file.is_open())
            {
                std::string line;
                std::getline(weights_file, line);
                dim0 = std::stoi(line);
                std::getline(weights_file, line);
                dim1 = std::stoi(line);
                while(std::getline(weights_file, line))
                {
                    weights.push_back(std::stof(line));
                }
                weights_file.close();
            }
            else
            {
                std::cout << "Unable to open file: " << weights_filename << std::endl;
            }

            bias_filename = bias_file_prefix + std::to_string(i) + bias_file_suffix;
            std::ifstream bias_file(bias_filename);
            if(bias_file.is_open())
            {
                std::string line;
                std::getline(bias_file, line);
                while(std::getline(bias_file, line))
                {
                    std::cout << std::stof(line) << std::endl;
                    bias.push_back(std::stof(line));
                }
                bias_file.close();
            }
            else
            {
                std::cout << "Unable to open file: " << bias_filename << std::endl;
            }
            // check the size of the network and create a Eigen matrix of the same size
            Eigen::MatrixXf weights_matrix(dim0, dim1);
            Eigen::MatrixXf bias_vector(1, dim1);
            for(int i = 0; i < dim0; ++i)
            {
                for(int j = 0; j < dim1; ++j)
                {
                    weights_matrix(i, j) = weights[i*dim1 + j];
                    if (i == 0 && j == 0)
                    {
                        std::cout << "weights[0][0] = " << weights_matrix(i, j) << std::endl;
                    }
                    if (i == dim0 - 1 && j == dim1 - 1)
                    {
                        std::cout << "weights[dim0-1][dim1-1] = " << weights_matrix(i, j) << std::endl;
                    }
                }
            }
            for(int i = 0; i < dim1; ++i)
            {
                bias_vector(0, i) = bias[i];
            }
            std::cout << "pushing back weights matrix with size" << weights_matrix.rows() << "x" << weights_matrix.cols() << std::endl;
            layer_weights.push_back(weights_matrix);
            std::cout << "pushing back bias vector" << std::endl;
            layer_bias.push_back(bias_vector);
        }
    }

    // implement ELU activation

    Eigen::MatrixXf ELU(Eigen::MatrixXf input)
    {
        Eigen::MatrixXf output = input;
        for (int i = 0; i < input.rows(); ++i)
        {
            for (int j = 0; j < input.cols(); ++j)
            {
                if (input(i, j) < 0)
                {
                    output(i, j) = exp(input(i, j)) - 1;
                }
            }
        }
        return output;
    }

    Eigen::MatrixXf forward(Eigen::MatrixXf input)
    {
        Eigen::MatrixXf output = input;
        for(int i = 0; i <=num_layers; ++i)
        {
            output = output * layer_weights[i] + layer_bias[i];
            // std::cout << "Output: " << output.size() << std::endl;
            if (i < num_layers)
            {
                output = ELU(output);
            }
        }
        return output;
    }
};


int main()
{
    NN_CPP nn(weights_file_prefix, weights_file_suffix, bias_file_prefix, bias_file_suffix, num_layers);
    std::cout << "LOADED" << std::endl;
    Eigen::MatrixXf input(1, 17);
    input = Eigen::MatrixXf::Zero(1, 17);

    Eigen::MatrixXf output = nn.forward(input);
    std::cout << "Output Size:" << output.size() << "Output:" << output << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        nn.forward(input);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time per inference: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000 << "us" << std::endl;
    return 0;
    
}