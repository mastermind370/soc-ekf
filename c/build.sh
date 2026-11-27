#!/bin/sh
# Build the C port without CMake (plain cc), into c/build/. Usage: sh c/build.sh
set -e
cd "$(dirname "$0")"
mkdir -p build
CC=${CC:-cc}
W="-std=c11 -Wall -Wextra -Wpedantic -Iinclude"
$CC $W -O2 tests/test_ekf_q.c src/ekf_q.c src/ekf_f.c -lm -o build/test_ekf_q
$CC $W -O2 tools/ekf_trace.c src/ekf_q.c src/ekf_f.c -lm -o build/ekf_trace
$CC $W -O2 -DEKF_COUNT_OPS tools/ekf_trace.c src/ekf_q.c src/ekf_f.c -lm -o build/ekf_trace_ops
$CC $W -O1 -g -fsanitize=undefined -fno-sanitize-recover=all tests/test_ekf_q.c src/ekf_q.c src/ekf_f.c -lm -o build/test_ekf_q_ubsan
echo "built: build/test_ekf_q build/test_ekf_q_ubsan build/ekf_trace build/ekf_trace_ops"
