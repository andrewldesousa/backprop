#include <iostream>
#include <memory>
#include <vector>
#include "../backprop.h"

// move operator
#include <utility>

// include exp
#include <cmath>


int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "debug") Logger::get_instance().set_debug_mode(true);
    else if (argc > 1) throw std::runtime_error("Invalid argument in main function. Use 'debug' to enable debug mode.");

    int num_epochs = 5000000, num_samples = 4;
    float learning_rate = 0.001;

    // weights shared pointer
    std::vector<std::shared_ptr<Scalar<double>>> weights{
        Scalar<double>::make(0),
        Scalar<double>::make(0.23),
        Scalar<double>::make(-.41)
    };

    std::shared_ptr<Scalar<double>> X[4][2];

    X[0][0] = Scalar<double>::make(0);
    X[0][1] = Scalar<double>::make(0);
    X[1][0] = Scalar<double>::make(0);
    X[1][1] = Scalar<double>::make(1);
    X[2][0] = Scalar<double>::make(1);
    X[2][1] = Scalar<double>::make(0);
    X[3][0] = Scalar<double>::make(1);
    X[3][1] = Scalar<double>::make(1);

    std::shared_ptr<Scalar<double>> Y[4];
    Y[0] = Scalar<double>::make(0);
    Y[1] = Scalar<double>::make(0);
    Y[2] = Scalar<double>::make(0);
    Y[3] = Scalar<double>::make(1);

    for (int i = 0; i < num_epochs; i++) {        
        std::shared_ptr<Scalar<double>> loss = Scalar<double>::make(0);
        for (int j = 0; j < num_samples; j++) {
            auto z = weights[0] * X[j][0] + weights[1] * X[j][1] + weights[2]; 
            auto a = sigmoid(z);
            // loss = cross_entropy(Y[j], a) + loss;
        }

        // loss = loss / Scalar<double>::make(num_samples);
        // loss->backward();
        
        // update weights
        for (int j = 0; j < weights.size(); j++) {
            weights[j]->value -= learning_rate * weights[j]->grad; 
            weights[j]->grad = 0;
        }

        if (i % 1000 == 0) Logger::get_instance().log(
            "Epoch: " + std::to_string(i) + "/Loss: " + std::to_string(loss->value),
            Logger::LogLevel::INFO
        );
    }

    return 0;
}