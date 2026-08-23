add_library(mlite_execution INTERFACE)
add_library(mlite::execution ALIAS mlite_execution)
set_target_properties(mlite_execution PROPERTIES EXPORT_NAME execution)

target_include_directories(
    mlite_execution
    INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(mlite_execution INTERFACE mlite_tensor)
target_compile_features(mlite_execution INTERFACE cxx_std_20)
