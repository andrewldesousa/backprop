#include <iostream>
#include "../backprop.h"
#include <memory>


int main() {
    // weights shared pointer
    std::shared_ptr<Scalar<double>> w1 = Scalar<double>::make(-.09);
    std::shared_ptr<Scalar<double>> w2 = Scalar<double>::make(.02);
    std::shared_ptr<Scalar<double>> b = Scalar<double>::make(0);

    int num_epochs = 10000, num_samples = 4;
    float learning_rate = 0.0001;
    auto& graph = ComputationalGraph<double>::get_instance();

    for (int i = 0; i < num_epochs; i++) {
        graph.clear();

        std::shared_ptr<Scalar<double>> X[4][2];
        X[0][0] = Scalar<double>::make(0);
        X[0][1] = Scalar<double>::make(0);
        X[1][0] = Scalar<double>::make(0);
        X[1][1] = Scalar<double>::make(1);
        X[2][0] = Scalar<double>::make(1);
        X[2][1] = Scalar<double>::make(0);
        X[3][0] = Scalar<double>::make(1);
        X[3][1] = Scalar<double>::make(1);

        // labels shared pointer
        std::shared_ptr<Scalar<double>> Y[4];
        Y[0] = Scalar<double>::make(0);
        Y[1] = Scalar<double>::make(0);
        Y[2] = Scalar<double>::make(0);
        Y[3] = Scalar<double>::make(1);

        
        std::shared_ptr<Scalar<double>> loss = std::make_shared<Scalar<double>>(0);
        for (int j = 0; j < num_samples; j++) {
            // forward
            auto z = w1 * X[j][0] + w2 * X[j][1] + b;
            auto a = sigmoid(z);
            loss = cross_entropy(Y[j], a) + loss;
        }

        loss->backward();
        
        // update weights
        w1->value = w1->value - learning_rate * w1->grad;
        w2->value = w2->value - learning_rate * w2->grad;
        b->value = b->value - learning_rate * b->grad;

        // print epoch loss
        if (i % 1000 == 0) {
            std::cout << "Epoch: " << i << " Loss: " << loss->value << std::endl;
        }
    }

    return 0;
}