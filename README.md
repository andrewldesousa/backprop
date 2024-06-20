# Backprop
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Unit Tests](https://github.com/andrewldesousa/backprop.cpp/actions/workflows/unit_tests.yaml/badge.svg)
Backprop is a library for automatically differentiation mathematical expressions.

## Usage
Add backprop.cpp as include in your project. Start by including the header file in your cmake file.
```cmake
include_directories("path/to/backprop")
```

Then include the header file in your source file.
```cpp
#include "backprop.h"
```

## Tests
```bash
mkdir build
cd build
cmake ..
make
./tests/test_main
```