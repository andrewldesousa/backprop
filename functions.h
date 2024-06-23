# pragma once
#include <iostream>
#include <memory>
#include "scalar.h"

// ce loss function leveraging shared
template <typename T>
std::shared_ptr<Scalar<T>> ce(std::shared_ptr<Scalar<T>> y, std::shared_ptr<Scalar<T>> y_hat) {
    if (y->value == 1) { return -log(y_hat); }
    else {
        auto one = std::make_shared<Scalar<T>>(1);
        return -log(one - y_hat);
    }
}

// sigmoid function leveraging shared
template <typename T>
std::shared_ptr<Scalar<T>> sigmoid(std::shared_ptr<Scalar<T>> x) {
    auto numerator = std::make_shared<Scalar<T>>(1);
    auto denominator = std::make_shared<Scalar<T>>(1) + exp(-x);

    // print numerator and denominator
    std::cout << "Numerator: " << numerator->value << std::endl;
    std::cout << "Denominator: " << denominator->value << std::endl;


    return numerator / denominator;
}