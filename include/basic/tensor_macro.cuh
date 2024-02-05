#pragma once

namespace dl{
  enum class Calculator{ADD, SUB, MUL, DIV, MOD};  // for calculator

  enum class Operator{SUM, MEAN, MAX, MIN};        // operator for Tensor

  enum class Axis{COL, ROW, CHANNEL};              // axis 0, 1, 2

  enum class Pool{MAX, AVG};

  enum class Device{CPU, CUDA};

  using f32 = float;
  using f64 = double;
  using i32 = int;
  using i64 = long long;

  const int BOOST_ROW = 1 << 2;
  const int BOOST_COL = 1 << 2;
  const int BOOST_CHANNEL = 1 << 2;
  const int BOOST_NUMBER = 1 << 2;
  #define NTHREAD_C(ncpu, num) (ncpu * 1 / num)
  #define NTHREAD_R(ncpu, num) (ncpu * 1 / num)

  // 循环分块的块大小
  const int BLOCK_SIZE = 64;

  // PRINT MACRO
  const int PRINT_PRECISION = 4;
  // if tensor col greater than it, then use ignored mode
  const int MAX_PRINT_COL = 25;    
  // if tensor row greater than it, then use ignored mode
  const int MAX_PRINT_ROW = 30;
  // if tensor channel greater than it, then use ignored mode
  const int MAX_PRINT_CHANNEL = 20;
  const int MAX_NUM_LINE = 6;      // max data/line

  const int SHOW_NUMBER_LEN = 3;

  const f32 eps = 1e-8;

  #define POOL_SIZE  128
}

