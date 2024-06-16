#include <iostream>
#include <set>
#include <functional>


template <typename T>
class Scalar {
public:
    T value;
    T grad = 0;

    // children set
    std::set<Scalar<T>*> children;
    int in_degrees = 0;
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
        Scalar<T>* result = new Scalar<T>(this->value + other.value);
        
        result->children.insert(this);
        result->children.insert(&other);
        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            // print result value and grad
            std::cout << "result value: " << result->value << ", grad: " << result->grad << std::endl;
            // delim 
            std::cout << "----------------" << std::endl;

            this->grad += 1.0 * result->grad;
            other.grad += 1.0 * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator-(Scalar<T>& other) {
        Scalar<T>* result = new Scalar<T>(this->value - other.value);
        
        result->children.insert(this);
        result->children.insert(&other);
        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            this->grad += 1.0 * result->grad;
            other.grad -= 1.0 * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator*(Scalar<T>& other) {
        Scalar<T>* result = new Scalar<T>(this->value * other.value);
        
        result->children.insert(this);
        result->children.insert(&other);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            this->grad += other.value * result->grad;
            other.grad += this->value * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator/(Scalar<T>& other) {
        Scalar<T>* result = new Scalar<T>(this->value / other.value);

        result->children.insert(this);
        result->children.insert(&other);
        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [&, result]() {
            this->grad += 1.0 / other.value * result->grad;
            other.grad -= this->value / (other.value * other.value) * result->grad;
        };

        return *result;
    }
};

int main(int argc, char* argv[]) {
    Scalar<float> s1(1.0);
    Scalar<float> s2(2.0);
    Scalar<float> s4(5.0);

    Scalar<float>& s3 = s1 + s2;
    Scalar<float>& s5 = s3 * s4;
    s5.grad = 1.0;
    s5.backward();
    return 0;
}
