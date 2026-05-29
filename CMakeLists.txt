cmake_minimum_required(VERSION 3.18)

project(
    mlite
    VERSION 0.0.5
    DESCRIPTION "ML library"
    LANGUAGES CXX
)

# =========================================
# C++ Standard
# =========================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =========================================
# Build Type
# =========================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# =========================================
# Compiler Flags
# =========================================

if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3)
endif()

# =========================================
# Find Dependencies
# =========================================

find_package(pybind11 REQUIRED)

# =========================================
# Python Extension Module
# =========================================

pybind11_add_module(
    _core

    cpp/bindings/module.cpp
    cpp/bindings/linear_models/py_linear_regression.cpp
    cpp/bindings/linear_models/py_logistic_regression.cpp

    cpp/src/linear_models/linear_regression.cpp
    cpp/src/linear_models/logistic_regression.cpp
)

# =========================================
# Include Directories
# =========================================

target_include_directories(
    _core
    PRIVATE
    cpp/include
)

# =========================================
# Compile Definitions
# =========================================

target_compile_definitions(
    _core
    PRIVATE
    MLITE_VERSION="${PROJECT_VERSION}"
)

# =========================================
# Install
# =========================================

install(
    TARGETS _core
    DESTINATION mlite
)