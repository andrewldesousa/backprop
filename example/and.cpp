#include <iostream>
#include "../scalar.h"


Scalar<double> cross_entropy(Scalar<double> y, Scalar<double> y_hat) {
    return -y * log(y_hat) - (1 - y) * log(1 - y_hat);
}

Scalar<double> sigmoid(Scalar<double> x) {
    return 1 / (1 + exp(-x));
}

int main() {
    // 4x2 dataset maxtrix
    Scalar<double> X[4][2] = {
        {Scalar<double>(0), Scalar<double>(0)},
        {Scalar<double>(0), Scalar<double>(1)},
        {Scalar<double>(1), Scalar<double>(0)},
        {Scalar<double>(1), Scalar<double>(1)}
    };

    Scalar<double> Y[4] = {
        Scalar<double>(0),
        Scalar<double>(0),
        Scalar<double>(0),
        Scalar<double>(1)
    };

    Scalar<double> w1(.1), w2(.1), b(.1);


    int num_epochs = 1000, num_samples = 4;
    for (int i = 0; i < num_epochs; i++) {
        Scalar<double> loss(0);
        
        for (int j = 0; j < num_samples; j++) {
            Scalar<double> z = X[j][0] * w1 + X[j][1] * w2 + b;
            Scalar<double> a = sigmoid(z);
            loss += cross_entropy(Y[j], a);
        }
        
        loss.backward();

        w1.value -= w1.grad * 0.1;
        w2.value -= w2.grad * 0.1;
        b.value -= b.grad * 0.1;

        w1.grad = 0;
        w2.grad = 0;
        b.grad = 0;
    }

    return 0;
}