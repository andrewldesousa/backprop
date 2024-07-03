# pragma once
#include <iostream>
#include <memory>
#include "scalar.h"

// ce loss function leveraging shared
template <typename T>
std::shared_ptr<Scalar<T>> cross_entropy(std::shared_ptr<Scalar<T>> y, std::shared_ptr<Scalar<T>> y_hat) {
    if (y->value == 1) { return -log(y_hat); }
    else {
        auto one = Scalar<T>::make(1);
        return -log(one - y_hat);
    }
}

// sigmoid function leveraging shared
template <typename T>
std::shared_ptr<Scalar<T>> sigmoid(std::shared_ptr<Scalar<T>> x) {
    auto negative_x = -x;
    auto exp_negative_x = exp(negative_x);    
    auto numerator = Scalar<T>::make(1);
    auto denominator_one = Scalar<T>::make(1);
    auto denominator = denominator_one + exp_negative_x;
    auto result = numerator / denominator;

    return result;
}

// mse loss function leveraging shared
template <typename T>
std::shared_ptr<Scalar<T>> mse(std::shared_ptr<Scalar<T>> y, std::shared_ptr<Scalar<T>> y_hat) {
    auto diff = y - y_hat;
    auto diff_squared = square(diff);
    return diff_squared;
}