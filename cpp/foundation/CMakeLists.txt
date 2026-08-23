add_library(
    mlite_foundation
    STATIC
    src/errors.cpp
)
add_library(mlite::foundation ALIAS mlite_foundation)
set_target_properties(mlite_foundation PROPERTIES EXPORT_NAME foundation)

set(MLITE_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MLITE_GENERATED_INCLUDE_DIR}/mlite/foundation")
configure_file(
    include/mlite/foundation/version.hpp.in
    "${MLITE_GENERATED_INCLUDE_DIR}/mlite/foundation/version.hpp"
    @ONLY
)

target_include_directories(
    mlite_foundation
    PUBLIC
    "$<BUILD_INTERFACE:${MLITE_GENERATED_INCLUDE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_features(mlite_foundation PUBLIC cxx_std_20)

if(MLITE_INSTALL_CXX)
    install(
        FILES "${MLITE_GENERATED_INCLUDE_DIR}/mlite/foundation/version.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlite/foundation"
        COMPONENT development
    )
endif()
