#pragma once
#include <iostream>
#include <string>
#include <filesystem>
#include <iostream>


class Logger {
    Logger() {} // Private Constructor
    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

public:
    enum class LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    bool debug_mode;

    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }

    void set_debug_mode(bool mode) { debug_mode = mode; }

    void log(const std::string& message, LogLevel level = LogLevel::INFO) {
        const std::string resetColor = "\033[0m";
        std::string colorCode;

        switch (level) {
            case LogLevel::DEBUG:
                colorCode = "\033[34m"; // Blue
                break;
            case LogLevel::INFO:
                colorCode = "\033[32m"; // Green
                break;
            case LogLevel::WARNING:
                colorCode = "\033[33m"; // Yellow
                break;
            case LogLevel::ERROR:
                colorCode = "\033[31m"; // Red
                break;
        }

        std::cout << colorCode << message << resetColor << std::endl;
    }
};

void write_graph_to_dot(std::string& filepath, std::shared_ptr<Scalar<double>> loss) {
    throw std::runtime_error("Not implemented");
}
