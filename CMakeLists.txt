cmake_minimum_required(VERSION 3.20)
project(ZeroGrad VERSION 0.1.0 LANGUAGES CXX)

# Enforcing C++20 and turning off compiler specific extensions
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ====== The C++ Setup ======
add_library(zerograd_core src/scalar.cpp)

# This is how CMake knows where the C++ headers reside
target_include_directories(zerograd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ====== Catch2 v3 (The Testing Framework) ======
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

enable_testing()

add_executable(zerograd_tests tests/test_scalar.cpp)

target_link_libraries(zerograd_tests PRIVATE Catch2::Catch2WithMain zerograd_core)

include(Catch)
catch_discover_tests(zerograd_tests)

# ====== PyBind11 Setup ======
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(zerograd_bindings bindings/bindings.cpp)

target_link_libraries(zerograd_bindings PRIVATE zerograd_core)

set_target_properties(zerograd_bindings PROPERTIES OUTPUT_NAME "_zerograd_backend")