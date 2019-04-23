#!/bin/bash

rm -rf build

cmake -Bbuild -H. \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DFLATBUFFERS_INCLUDE_DIR=$HOME/work/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include \
  -DTFLITE_INCLUDE_DIR=$HOME/work/tensorflow/ \
  -DTFLITE_LIBRARY_DIR=$HOME/work/tensorflow/tensorflow/lite/tools/make/build
