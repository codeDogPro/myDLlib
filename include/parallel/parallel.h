#pragma once

#include <parallel/thread_pool.h>
#include <parallel/thread_calcu.h>

const int pool_size = 100;

namespace dl{
  template<typename T, typename Func, typename... Types>
  void
  parallel(Func func, int nthread, Types &...args){
    for(int t = 0; t < nthread; t++){
      std::thread task(func<T>, args...);
    }
  }
}