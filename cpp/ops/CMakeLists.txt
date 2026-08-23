add_library(
    mlite_ops
    STATIC
    src/arithmetic.cpp
    src/creation.cpp
    src/indexing.cpp
    src/tensor_iterator.cpp
)
add_library(mlite::ops ALIAS mlite_ops)
set_target_properties(mlite_ops PROPERTIES EXPORT_NAME ops)

target_include_directories(
    mlite_ops
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(mlite_ops PUBLIC mlite_execution)
target_compile_features(mlite_ops PUBLIC cxx_std_20)
