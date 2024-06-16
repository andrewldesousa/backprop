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
    int in_degrees;
    std::function<void()> _backward;


    Scalar(T value) : value(value), in_degrees(0) {
        _backward = []() {};
    }

    void backward() {
        // if in_degrees is not zero, return
        if (in_degrees != 0) { throw std::runtime_error("in_degrees is not zero"); }

        grad = 1.0;
        _backward();

        // backward children
        for (auto child : children) {
            child->in_degrees--;
            
            if (child->in_degrees == 0) {
                child->_backward();
            }
        }

    }

    Scalar<T>& operator+(Scalar<T>& other) {
        Scalar<T>* result = new Scalar<T>(this->value + other.value);
        this->children.insert(result);
        other.children.insert(result);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [this, &other, &result]() {
            this->grad += 1.0 * result->grad;
            other.grad += 1.0 * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator-(Scalar<T>& other) {
        Scalar<T>* result = new Scalar<T>(this->value - other.value);
        this->children.insert(result);
        other.children.insert(result);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [this, &other, &result]() {
            this->grad += 1.0 * result->grad;
            other.grad -= 1.0 * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator*(Scalar<T>& other) {
        Scalar<T>* result = new Scalar<T>(this->value * other.value);
        this->children.insert(result);
        other.children.insert(result);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [this, &other, &result]() {
            this->grad += other.value * result->grad;
            other.grad += this->value * result->grad;
        };

        return *result;
    }

    Scalar<T>& operator/(Scalar<T>& other) {
        Scalar<T>* result = new Scalar<T>(this->value / other.value);
        this->children.insert(result);
        other.children.insert(result);

        this->in_degrees++;
        other.in_degrees++;

        result->_backward = [this, &other, &result]() {
            this->grad += 1.0 / other.value * result->grad;
            other.grad -= this->value / (other.value * other.value) * result->grad;
        };

        return *result;
    }
};

int main(int argc, char* argv[]) {
    Scalar<float> s1(1.0);
    Scalar<float> s2(2.0);

    Scalar<float>& s3 = s1 + s2;
    Scalar<float>& s4 = s1 - s2;
    Scalar<float>& s5 = s1 * s2;

    s5.backward(), s4.backward(), s3.backward();

    std::cout << "s1.grad: " << s1.grad << std::endl;
    std::cout << "s2.grad: " << s2.grad << std::endl;


    return 0;
}
