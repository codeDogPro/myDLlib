#pragma once

namespace dl{
  enum{PLUS, MINUS, MULTIPLY, DIVIDE, MOD};          // for calculator

  enum{SUM, MEAN};                                   // operate tensor

  enum{NvsN, Nvs1};

  // enum{ARTHMATIC_F, ARTHMATIC_S, SUMUP, AVERAGE, GETMAX, GETMIN};

  const int BOOST_ROW = 128;
  const int BOOST_CHANNEL = 256;
  #define NTHREAD_C(ncpu) ncpu * 1.5
  #define NTHREAD_R(ncpu) ncpu * 1
}
