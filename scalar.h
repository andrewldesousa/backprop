#pragma once

#include <iostream>
#include <set>
#include <functional>
#include <memory>


template <typename T>
class Scalar: public std::enable_shared_from_this<Scalar<T>> {
public:
    T value;
    T grad = 0;
    int in_degrees = 0;

    std::set<std::shared_ptr<Scalar>> children;
    std::function<void()> _backward;

    Scalar(T value) : value(value), in_degrees(0) {
        _backward = []() {};
    }

    void backward(bool is_root=true) {
        // if in_degrees is not zero, return
        if (in_degrees != 0) { throw std::runtime_error("in_degrees is not zero"); }
        if (is_root) { this->grad = 1.0; }
        
        _backward();

        // backward children
        for (auto child : children) {
            child->in_degrees--;
            if (child->in_degrees == 0) { child->backward(false); }
        }
    }

    friend std::shared_ptr<Scalar<T>> operator+(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = std::make_shared<Scalar<T>>(lhs->value + rhs->value);
        
        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [&, result]() {
            lhs->grad += 1.0 * result->grad;
            rhs->grad += 1.0 * result->grad;
        };

        return result;
    }

    friend std::shared_ptr<Scalar<T>> operator-(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = std::make_shared<Scalar<T>>(lhs->value - rhs->value);

        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [&, result]() {
            lhs->grad += 1.0 * result->grad;
            rhs->grad += -1.0 * result->grad;
        };

        return result;
    }

    friend std::shared_ptr<Scalar<T>> operator*(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = std::make_shared<Scalar<T>>(lhs->value * rhs->value);

        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [&, result]() {
            lhs->grad += rhs->value * result->grad;
            rhs->grad += lhs->value * result->grad;
        };

        return result;
    }

    friend std::shared_ptr<Scalar<T>> operator/(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = std::make_shared<Scalar<T>>(lhs->value / rhs->value);

        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [&, result]() {
            lhs->grad += 1.0 / rhs->value * result->grad;
            rhs->grad += -lhs->value / (rhs->value * rhs->value) * result->grad;
        };

        return result;
    }

    // // unary operators
    friend std::shared_ptr<Scalar<T>> operator+(std::shared_ptr<Scalar<T>> rhs) {
        return rhs;
    }

    friend std::shared_ptr<Scalar<T>> operator-(std::shared_ptr<Scalar<T>> rhs) {
        auto result = std::make_shared<Scalar<T>>(-rhs->value);

        result->children.insert(rhs);
        rhs->in_degrees++;

        result->_backward = [&, result]() {
            rhs->grad += -1.0 * result->grad;
        };

        return result;
    }

    // exp
    friend std::shared_ptr<Scalar<T>> exp(std::shared_ptr<Scalar<T>> rhs) {
        auto result = std::make_shared<Scalar<T>>(std::exp(rhs->value));

        result->children.insert(rhs);
        rhs->in_degrees++;

        result->_backward = [&, result]() {
            rhs->grad += std::exp(rhs->value) * result->grad;
        };

        return result;
    }

    // log
    friend std::shared_ptr<Scalar<T>> log(std::shared_ptr<Scalar<T>> rhs) {
        auto result = std::make_shared<Scalar<T>>(std::log(rhs->value));

        result->children.insert(rhs);
        rhs->in_degrees++;

        result->_backward = [&, result]() {
            rhs->grad += 1.0 / rhs->value * result->grad;
        };

        return result;
    }
};