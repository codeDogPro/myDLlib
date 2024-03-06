#!/bin/bash

if [ "$1" == "rebuild" ]; then
  cmake -B build && cmake --build build && sudo nvprof ./build/cuda_test
else
  cmake --build build && sudo nvprof ./build/cuda_test
fi