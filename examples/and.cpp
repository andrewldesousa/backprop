#include <iostream>
#include "../backprop.h"
#include <memory>


int main() {
    // 4x2 dataset maxtrix shared pointer
    std::shared_ptr<Scalar<double>> X[4][2];
    X[0][0] = std::make_shared<Scalar<double>>(0);
    X[0][1] = std::make_shared<Scalar<double>>(0);
    X[1][0] = std::make_shared<Scalar<double>>(0);
    X[1][1] = std::make_shared<Scalar<double>>(1);
    X[2][0] = std::make_shared<Scalar<double>>(1);
    X[2][1] = std::make_shared<Scalar<double>>(0);
    X[3][0] = std::make_shared<Scalar<double>>(1);
    X[3][1] = std::make_shared<Scalar<double>>(1);

    // labels shared pointer
    std::shared_ptr<Scalar<double>> Y[4];
    Y[0] = std::make_shared<Scalar<double>>(0);
    Y[1] = std::make_shared<Scalar<double>>(0);
    Y[2] = std::make_shared<Scalar<double>>(0);
    Y[3] = std::make_shared<Scalar<double>>(1);

    // weights shared pointer
    std::shared_ptr<Scalar<double>> w1 = std::make_shared<Scalar<double>>(-.09);
    std::shared_ptr<Scalar<double>> w2 = std::make_shared<Scalar<double>>(.02);
    std::shared_ptr<Scalar<double>> b = std::make_shared<Scalar<double>>(0);
    int num_epochs = 10000, num_samples = 4;
    float learning_rate = 0.001;

    for (int i = 0; i < num_epochs; i++) {
        // loss
        std::shared_ptr<Scalar<double>> loss = std::make_shared<Scalar<double>>(0);

        for (int j = 0; j < num_samples; j++) {
            // forward
            auto z = w1 * X[j][0];
            loss = z;

            break;
        }

        std::cout << "Epoch: " << i << " Loss: " << loss->value << std::endl;
        loss->backward();

        // print updating weights
        
        // update weights
        w1->value = w1->value - learning_rate * w1->grad;
        w2->value = w2->value - learning_rate * w2->grad;
        b->value = b->value - learning_rate * b->grad;

        // print grads
        // printing

        std::cout << "Printing grads:\n";

        std::cout << "w1 grad: " << w1->grad << std::endl;
        std::cout << "w2 grad: " << w2->grad << std::endl;
        std::cout << "b grad: " << b->grad << std::endl;

        // reset gradients
        w1->grad = 0;
        w2->grad = 0;
        b->grad = 0;
    }

    return 0;
}