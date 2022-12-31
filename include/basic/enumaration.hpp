#pragma once

namespace dl{
  enum{PLUS, MINUS, MULTIPLY, DIVIDE, MOD};  // for calculator

  enum{SUM, MEAN, MAX, MIN};                 // operator for Tensor

  enum{NvsN, Nvs1};

  enum{COL, ROW, CHANNEL};                   // axis 0, 1, 2

  // enum{ARTHMATIC_F, ARTHMATIC_S, SUMUP, AVERAGE, GETMAX, GETMIN};

  const int BOOST_ROW = 1 << 7;
  const int BOOST_CHANNEL = 1 << 0;
  #define NTHREAD_C(ncpu) (ncpu * 1)
  #define NTHREAD_R(ncpu) (ncpu * 1)
}
