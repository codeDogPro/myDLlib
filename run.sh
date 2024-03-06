#!/bin/bash

if [ "$1" == "reconfig" ]; then
  cmake -B build && cmake --build build && ./build/cuda_test
else
  cmake --build build && ./build/cuda_test
fi