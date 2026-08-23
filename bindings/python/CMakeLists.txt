pybind11_add_module(
    _core
    MODULE
    src/module.cpp
    src/numpy.cpp
    src/tensor.cpp
)

target_include_directories(
    _core
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    _core
    PRIVATE
    mlite_ops
    mlite_backend_cpu
)

target_compile_features(_core PRIVATE cxx_std_20)
mlite_configure_target(_core)

install(TARGETS _core DESTINATION mlite COMPONENT python)
