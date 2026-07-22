cmake_minimum_required(VERSION 3.18)

project(
    mlite
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

file(GLOB_RECURSE BINDINGS
    CONFIGURE_DEPENDS
    core/bindings/*.cpp
)

pybind11_add_module(
    _core

    ${BINDINGS}

    core/src/linear_models/linear_regression.cpp
    core/src/linear_models/logistic_regression.cpp

    core/src/model_selection/train_test_split.cpp
    core/src/model_selection/stratified_shuffle_split.cpp
    core/src/model_selection/k_fold.cpp

    core/src/metrics/r2_score.cpp
    core/src/metrics/accuracy_score.cpp
    core/src/metrics/mean_squared_error.cpp
    core/src/metrics/root_mean_squared_error.cpp
    core/src/metrics/mean_absolute_error.cpp
    core/src/metrics/recall_score.cpp

    core/src/preprocessing/one_hot_encoder.cpp
    core/src/preprocessing/ordinal_encoder.cpp
)

# =========================================
# Include Directories
# =========================================

target_include_directories(
    _core
    PRIVATE
    core/include
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