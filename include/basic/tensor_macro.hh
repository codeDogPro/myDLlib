#pragma once

namespace dl{
  enum class Calculator{PLUS, MINUS, MULTIPLY, DIVIDE, MOD};  // for calculator

  enum class Operator{SUM, MEAN, MAX, MIN};                   // operator for Tensor

  enum class Axis{COL, ROW, CHANNEL};                         // axis 0, 1, 2

  enum class Pool{MAX, AVG};

  using f32 = float;
  using f64 = double;
  using i32 = int;
  using i64 = long long;

  const int BOOST_ROW = 1 << 0;
  const int BOOST_COL = 1 << 0;
  const int BOOST_CHANNEL = 1 << 0;
  #define NTHREAD_C(ncpu, num) (ncpu * 1 / num)
  #define NTHREAD_R(ncpu, num) (ncpu * 1 / num)

  const int PRINT_PRECISION = 4;

  const float eps = 1e-8;

  #define POOL_SIZE  128
}

