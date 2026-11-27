cmake_minimum_required(VERSION 3.16)
project(soc_ekf_c C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)
include_directories(include)

add_library(ekf STATIC src/ekf_q.c src/ekf_f.c)
target_link_libraries(ekf m)

add_executable(test_ekf_q tests/test_ekf_q.c)
target_link_libraries(test_ekf_q ekf)

add_executable(ekf_trace tools/ekf_trace.c)
target_link_libraries(ekf_trace ekf)

# op-counting build for the Cortex-M4 cycle estimate
add_executable(ekf_trace_ops tools/ekf_trace.c src/ekf_q.c src/ekf_f.c)
target_compile_definitions(ekf_trace_ops PRIVATE EKF_COUNT_OPS)
target_link_libraries(ekf_trace_ops m)

# sanitizer build of the tests: any signed overflow / UB aborts
add_executable(test_ekf_q_ubsan tests/test_ekf_q.c src/ekf_q.c src/ekf_f.c)
target_compile_options(test_ekf_q_ubsan PRIVATE -O1 -g -fsanitize=undefined -fno-sanitize-recover=all)
target_link_options(test_ekf_q_ubsan PRIVATE -fsanitize=undefined)
target_link_libraries(test_ekf_q_ubsan m)

enable_testing()
add_test(NAME ekf_q COMMAND test_ekf_q)
add_test(NAME ekf_q_ubsan COMMAND test_ekf_q_ubsan)
