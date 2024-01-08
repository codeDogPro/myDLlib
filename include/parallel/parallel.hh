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
    nthread = cpu_number();
    _M_pool.start(nthread);
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_row(Fn&& f, std::shared_ptr<Tensor<Tp>> output, Ts&&... cargs) {
    int output_row = output->row();
    int task_size = (int)std::ceil((double)output_row / nthread);

    for(int task_begin = 0; task_begin < output_row; task_begin += task_size){
      auto ret = _M_pool.submit([task_begin, task_size, &f, &output, &cargs...]
        {return std::forward<Fn>(f) 
          (task_begin, task_size, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    __sync_rets();
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_col(Fn&& f, std::shared_ptr<Tensor<Tp>> output, Ts&&... cargs) {
    int output_col = output->col();
    int task_size = (int)std::ceil((double)output_col / nthread);

    for(int task_begin = 0; task_begin < output_col; task_begin += task_size){
      auto ret = _M_pool.submit([task_begin, task_size, &f, &output, &cargs...]
        {return std::forward<Fn>(f) 
          (task_begin, task_size, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    __sync_rets();
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_channel(Fn&& f, std::shared_ptr<Tensor<Tp>> output, Ts&&... cargs) {
    int output_ch = output->channel();
    int task_size = (int)std::ceil((double)output_ch / nthread);
    // printf("task_size: %d\n", task_size);
    // printf("output_ch: %d\n", output_ch);
    // printf("In channel output use_count: %ld\n", output.use_count());

    for(int task_begin = 0; task_begin < output_ch; task_begin += task_size){
      auto ret = _M_pool.submit([task_begin, task_size, &f, &output, &cargs...]
        {return std::forward<Fn>(f) 
          (task_begin, task_size, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    __sync_rets();
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_number(Fn&& f, std::shared_ptr<Tensor<Tp>> output, Ts&&... cargs) {
    int output_num = output->number();
    int task_size = (int)std::ceil((double)output_num / nthread);

    for(int task_begin = 0; task_begin < output_num; task_begin += task_size){
      auto ret = _M_pool.submit([task_begin, task_size, &f, &output, &cargs...]
        {return std::forward<Fn>(f) 
          (task_begin, task_size, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    __sync_rets();
  }

private:
  void __sync_rets(){
    for(int i = rets.size() - 1; i >= 0; i--){
      rets[i].get();
      rets.pop_back();
    }
  }

private:
  thread_pool _M_pool;
  std::vector<std::future<bool>> rets;
  size_t nthread;
};

static Parallelizer parallelizer(POOL_SIZE);

} // namespace dl