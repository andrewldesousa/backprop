#pragma once

#include <iostream>
#include <set>
#include <functional>
#include <memory>
#include <fstream>
#include <filesystem>
#include "utils.h"

static int id_counter;

// forward declaration
template <typename T>
class Scalar;

template <typename T>
class ComputationalGraph {
    // Make the constructor private to enforce singleton pattern
    ComputationalGraph() {}

public:
    std::set<std::shared_ptr<Scalar<T>>> nodes;
    std::shared_ptr<Scalar<T>> root;

    ComputationalGraph(const ComputationalGraph&) = delete;
    ComputationalGraph& operator=(const ComputationalGraph&) = delete;

    // Static method to access the singleton instance
    static ComputationalGraph<T>& get_instance() {
        static ComputationalGraph<T> instance;
        return instance;
    }

    void add_node(std::shared_ptr<Scalar<T>> node) { nodes.insert(node); }

    void clear() { 
        for (auto node : nodes) {
            node->grad = 0;
            node->in_degrees = 0;
            node->children.clear();
        }

        nodes.clear(); 
        root = NULL;
    }

    void write_dot(std::string& filename) {
        // create subdirs from filename
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream file;
        
        file.open(filename);
        if (!file.is_open()) 
            throw std::runtime_error("Could not open file: " + filename);

        file << "digraph G {\n";
        for (auto node : nodes) file << "  " << node.get()->to_string() << "\n";

        for (auto node : nodes) 
            for (auto child : node->children) 
                file << "  node" << node->id << " -> node" << child->id << "\n";

        file << "}\n";
        file.close();

        if (Logger::get_instance().debug_mode) 
            Logger::get_instance().log("Wrote graph to file: " + filename);
    }
};

template <typename T>
class Scalar: public std::enable_shared_from_this<Scalar<T>> {
    
    void add_child(std::shared_ptr<Scalar<T>> child) {
        children.insert(child);
        child->in_degrees++;
    }

public:
    const int id = id_counter++;
    T value, grad = 0;
    int in_degrees = 0;

    std::set<std::shared_ptr<Scalar<T>>> children;
    std::function<void()> _backward;

    Scalar() : Scalar(0) {}
    Scalar(T value) : value{value}, in_degrees{0} { _backward = []() {}; }

    static std::shared_ptr<Scalar<T>> make(T value) {
        auto s = std::make_shared<Scalar<T>>(value);
        ComputationalGraph<T>::get_instance().add_node(s);
        return s;
    }

    void backward(bool is_root=true) {
        // if in_degrees is not zero, return
        if (in_degrees != 0) { throw std::runtime_error("in_degrees is not zero"); }
        if (is_root) this->grad = 1.0;

        _backward();

        // backward children
        for (auto child : children) {
            child->in_degrees--;
            if (child->in_degrees == 0)
                child->backward(false);
        }
    }

    friend std::shared_ptr<Scalar<T>> operator+(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(lhs->value + rhs->value);

        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [lhs, rhs, result]() {
            lhs->grad += 1.0 * result->grad;
            rhs->grad += 1.0 * result->grad;
        };

        return result;
    }

    friend std::shared_ptr<Scalar<T>> operator-(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(lhs->value - rhs->value);

        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [lhs, rhs, result]() {
            lhs->grad += 1.0 * result->grad;
            rhs->grad += -1.0 * result->grad;
        };

        return result;
    }

    friend std::shared_ptr<Scalar<T>> operator*(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(lhs->value * rhs->value);

        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [lhs, rhs, result]() {
            lhs->grad += rhs->value * result->grad;
            rhs->grad += lhs->value * result->grad;
        };

        return result;
    }

    friend std::shared_ptr<Scalar<T>> operator/(std::shared_ptr<Scalar<T>> lhs, std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(lhs->value / rhs->value);

        result->children.insert(lhs);
        result->children.insert(rhs);
        lhs->in_degrees++;
        rhs->in_degrees++;

        result->_backward = [lhs, rhs, result]() {
            lhs->grad += 1.0 / rhs->value * result->grad;
            rhs->grad += (-1.0 /(rhs->value * rhs->value)) * result->grad;
        };

        return result;
    }

    // unary operators
    friend std::shared_ptr<Scalar<T>> operator-(std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(-rhs->value);

        result->children.insert(rhs);
        rhs->in_degrees++;

        result->_backward = [rhs, result]() {
            rhs->grad += -1.0 * result->grad;
        };

        return result;
    }

    friend std::shared_ptr<Scalar<T>> operator+(std::shared_ptr<Scalar<T>> rhs) {
        return rhs;
    }

    friend std::shared_ptr<Scalar<T>> exp(std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(std::exp(rhs->value));

        result->children.insert(rhs);
        rhs->in_degrees++;

        result->_backward = [rhs, result]() {
            rhs->grad += std::exp(rhs->value) * result->grad;
        };

        return result;
    }

    // log
    friend std::shared_ptr<Scalar<T>> log(std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(std::log(rhs->value));

        result->children.insert(rhs);
        rhs->in_degrees++;

        result->_backward = [rhs, result]() {
            rhs->grad += 1.0 / rhs->value * result->grad;
        };

        return result;
    }

    // square operator
    friend std::shared_ptr<Scalar<T>> square(std::shared_ptr<Scalar<T>> rhs) {
        auto result = Scalar<T>::make(rhs->value * rhs->value);
        result->children.insert(rhs);
        rhs->in_degrees++;

        result->_backward = [rhs, result]() {
            rhs->grad += 2.0 * rhs->value * result->grad;
        };

        return result;
    }

    // dont allow inplace operations
    Scalar<T>& operator+=(const Scalar<T>& rhs) = delete;
    Scalar<T>& operator-=(const Scalar<T>& rhs) = delete;
    Scalar<T>& operator*=(const Scalar<T>& rhs) = delete;
    Scalar<T>& operator/=(const Scalar<T>& rhs) = delete;

    std::string to_string() {
        std::string node_str;

        // "ID: 1\nAddress: 0x7ffdb2c2d590\nValue: 5.0\nGrad: 0.0"];
        node_str += "node" + std::to_string(this->id) + " [label=\"";
        node_str += "ID: " + std::to_string(this->id) + "\\n";
        node_str += "Value: " + std::to_string(this->value) + "\\n";
        node_str += "Grad: " + std::to_string(this->grad) + "\"]";

        return node_str;
    }
};
