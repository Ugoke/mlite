add_library(
    mlite_tensor
    STATIC
    src/dtype.cpp
    src/shape.cpp
    src/storage.cpp
    src/tensor.cpp
    src/tensor_view.cpp
)
add_library(mlite::tensor ALIAS mlite_tensor)
set_target_properties(mlite_tensor PROPERTIES EXPORT_NAME tensor)

target_include_directories(
    mlite_tensor
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(mlite_tensor PUBLIC mlite_foundation)
target_compile_features(mlite_tensor PUBLIC cxx_std_20)
