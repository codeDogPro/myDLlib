#pragma once

// #define BENCH
#ifdef BENCH
#include <basic/timer.hh>
#endif

#include "thread_pool.hh"
#include <basic/tensor_macro.hh>

#include <iostream>

namespace dl{

static thread_pool _M_pool{POOL_SIZE};

static size_t 
cpu_number() noexcept {
  return std::thread::hardware_concurrency();
}

static void 
thread_pool_init(){
  size_t num = cpu_number(); 
  _M_pool.start(num);
}

template<typename Fn, typename Tp, typename... Ts>
void parallel_4D(Fn&& f, std::shared_ptr<Tensor<Tp>> output, Ts&&... cargs) {
  size_t nthread = cpu_number(), output_ch = output->channel();
  size_t task_size = output_ch / nthread;
  std::vector<std::future<bool>> rets;
  for(size_t i = 0; i < nthread; i++){
    size_t begin_ch = i * task_size;
    auto ret = _M_pool.submit([&] {return std::forward<Fn>(f)
                  (begin_ch, task_size, output, std::cref(cargs)...);});
    rets.emplace_back(std::move(ret));
  }
  for(auto &ret : rets){
    ret.get();
    // std::cout << ret.get() << std::endl;
  }
}

} // namespace dl