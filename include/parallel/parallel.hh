#pragma once

// #define BENCH
#ifdef BENCH
#include <basic/timer.hh>
#endif

#include "thread_pool.hh"
#include <basic/tensor_macro.hh>

#include <cmath>
#include <iostream>

namespace dl{

static size_t 
cpu_number() noexcept {
  return std::thread::hardware_concurrency();
}

class Parallelizer {
public:
  Parallelizer();
  Parallelizer(size_t pool_size) : _M_pool(pool_size) {
    size_t ncpu = cpu_number(); 
    _M_pool.start(ncpu);
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_channel(Fn&& f, std::shared_ptr<Tensor<Tp>> output, Ts&&... cargs) {
    size_t nthread = cpu_number();
    int output_ch = output->channel();
    int task_size = (int)std::ceil((double)output_ch / nthread);
    // printf("task_size: %ld\n", task_size);
    // printf("output_ch: %ld\n", output_ch);
    // printf("In 4D output use_count: %ld\n", output.use_count());

    std::vector<std::future<bool>> rets;
    for(int task_begin = 0; task_begin < output_ch; task_begin += task_size){
      auto ret = _M_pool.submit([task_begin, task_size, &f, &output, &cargs...]
        {return std::forward<Fn>(f) 
          (task_begin, task_size, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    for(auto &ret : rets){
      ret.get();
      // std::cout << ret.get() << std::endl;
    }
  }

private:
  thread_pool _M_pool;
};

static Parallelizer parallelizer(POOL_SIZE);

} // namespace dl