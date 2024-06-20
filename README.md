# Backprop
<p align="center">
    <img src="static/cartoon.webp" alt="Image" width="512" height="512">
</p>

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