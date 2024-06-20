#include <iostream>
#include <set>
#include <functional>
#include <memory>


template <typename T>
class Scalar {
public:
    T value;
    T grad = 0;
    int in_degrees = 0;

    std::set<std::shared_ptr<Scalar>> children;
    std::function<void()> _backward;

    Scalar(T value) : value(value), in_degrees(0) {
        _backward = []() {};
    }

    void backward() {
        // if in_degrees is not zero, return
        if (in_degrees != 0) { throw std::runtime_error("in_degrees is not zero"); }
        _backward();

        // backward children
        for (auto child : children) {
            child->in_degrees--;
            if (child->in_degrees == 0) { child->backward(); }
        }
    }

    Scalar<T>& operator+(Scalar<T>& other) {
        auto result = std::make_shared<Scalar<T>>(this->value + other.value);
        
        auto other_shared_ptr = std::shared_ptr<Scalar<T>>(&other);
        auto this_shared_ptr = std::shared_ptr<Scalar<T>>(this);
        result->children.insert(this_shared_ptr);
        result->children.insert(other_shared_ptr);
        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            this->grad += 1.0 * result->grad;
            other.grad += 1.0 * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator-(Scalar<T>& other) {
        auto result = std::make_shared<Scalar<T>>(this->value - other.value);
        
        auto other_shared_ptr = std::shared_ptr<Scalar<T>>(&other);
        auto this_shared_ptr = std::shared_ptr<Scalar<T>>(this);
        result->children.insert(this_shared_ptr);
        result->children.insert(other_shared_ptr);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            this->grad += 1.0 * result->grad;
            other.grad -= 1.0 * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator*(Scalar<T>& other) {
        auto result = std::make_shared<Scalar<T>>(this->value * other.value);
        
        auto other_shared_ptr = std::shared_ptr<Scalar<T>>(&other);
        auto this_shared_ptr = std::shared_ptr<Scalar<T>>(this);
        result->children.insert(this_shared_ptr);
        result->children.insert(other_shared_ptr);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            this->grad += other.value * result->grad;
            other.grad += this->value * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator/(Scalar<T>& other) {
        auto result = std::make_shared<Scalar<T>>(this->value / other.value);

        auto other_shared_ptr = std::shared_ptr<Scalar<T>>(&other);
        auto this_shared_ptr = std::shared_ptr<Scalar<T>>(this);
        result->children.insert(this_shared_ptr);
        result->children.insert(other_shared_ptr);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            this->grad += 1.0 / other.value * result->grad;
            other.grad -= this->value / (other.value * other.value) * result->grad;
        };

        return *result;
    }
};


// numerical and scalar overload operators
template <typename T>
Scalar<T>& operator+(Scalar<T>& s1, T s2) {
    auto result = std::make_shared<Scalar<T>>(s1.value + s2);
    auto s1_shared_ptr = std::shared_ptr<Scalar<T>>(&s1);
    result->children.insert(s1_shared_ptr);
    s1.in_degrees++;

    result->_backward = [&, result]() {
        s1.grad += 1.0 * result->grad;
    };

    return *result;
}

// other order
template <typename T>
Scalar<T>& operator+(T s1, Scalar<T>& s2) {
    return s2 + s1;
}

template <typename T>
Scalar<T>& operator-(Scalar<T>& s1, T s2) {
    auto result = std::make_shared<Scalar<T>>(s1.value - s2);
    auto s1_shared_ptr = std::shared_ptr<Scalar<T>>(&s1);
    result->children.insert(s1_shared_ptr);
    s1.in_degrees++;

    result->_backward = [&, result]() {
        s1.grad += 1.0 * result->grad;
    };

    return *result;
}

// other order
template <typename T>
Scalar<T>& operator-(T s1, Scalar<T>& s2) {
    return s2 - s1;
}

template <typename T>
Scalar<T>& operator*(Scalar<T>& s1, T s2) {
    auto result = std::make_shared<Scalar<T>>(s1.value * s2);
    auto s1_shared_ptr = std::shared_ptr<Scalar<T>>(&s1);
    result->children.insert(s1_shared_ptr);
    s1.in_degrees++;

    result->_backward = [&, result]() {
        s1.grad += s2 * result->grad;
    };

    return *result;
}

template <typename T>
Scalar<T>& operator*(T s1, Scalar<T>& s2) {
    return s2 * s1;
}

template <typename T>
Scalar<T>& operator/(Scalar<T>& s1, T s2) {
    auto result = std::make_shared<Scalar<T>>(s1.value / s2);
    auto s1_shared_ptr = std::shared_ptr<Scalar<T>>(&s1);
    result->children.insert(s1_shared_ptr);
    s1.in_degrees++;

    result->_backward = [&, result]() {
        s1.grad += 1.0 / s2 * result->grad;
    };

    return *result;
}

template <typename T>
Scalar<T>& operator/(T s1, Scalar<T>& s2) {
    return s2 / s1;
}
