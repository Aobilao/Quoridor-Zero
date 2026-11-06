#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
PY="${PY:-python}"
FLAGS="-O3 -march=native -std=c++17 -Wall -Wextra"
SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
INCLUDES=$($PY -m pybind11 --includes)
mkdir -p build
printf '%s\n' -std=c++17 -Wall -Wextra $INCLUDES > cpp/compile_flags.txt
$CXX $FLAGS cpp/test_engine.cpp -o build/test_engine
$CXX $FLAGS -shared -fPIC $INCLUDES cpp/bindings.cpp -o "quoridor/quoridor_engine$SUFFIX"
echo "built quoridor/quoridor_engine$SUFFIX for $($PY --version)"
