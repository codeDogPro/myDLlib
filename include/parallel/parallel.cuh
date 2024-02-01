#pragma once

// #define BENCH
#ifdef BENCH
#include <basic/timer.cuh>
#endif

#include "thread_pool.cuh"
#include <basic/tensor_macro.cuh>

#include <iostream>

namespace dl{

static size_t 
cpu_number() noexcept {
  return std::thread::hardware_concurrency();
}

class Parallelizer {
public:
  Parallelizer() = default;

  Parallelizer(size_t pool_size) : _M_pool(pool_size) {
    nthread = cpu_number();
    _M_pool.start(nthread);
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_row(Fn&& fn, std::shared_ptr<Tensor<Tp>> output, 
  int offset, Ts&&... cargs) {
    int output_row = output->row(), col = output->col();
    int task_size = (output_row / nthread) ? output_row / nthread : 1;
    int align_end = output_row - output_row % task_size;

    for(int task_begin = 0; task_begin < align_end; task_begin += task_size){
      auto ret = _M_pool.submit(
        [task_begin, task_size, col, offset, &fn, &output, &cargs...]
        {return std::forward<Fn>(fn) 
          (task_begin, task_size, col, offset, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    // rest of tasks
    int task_begin = align_end, task_sz = output_row % task_size;
    auto ret = _M_pool.submit(
      [task_begin, task_sz, col, offset, &fn, &output, &cargs...]
      {return std::forward<Fn>(fn) 
        (task_begin, task_sz, col, offset, output, std::cref(cargs)...);
      });
    rets.push_back(std::move(ret));
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_col(Fn&& fn, std::shared_ptr<Tensor<Tp>> output,
  int offset, Ts&&... cargs) {
    int output_col = output->col(), row = output->row();
    int task_size = (output_col / nthread) ? output_col / nthread : 1;
    int align_end = output_col - output_col % task_size;

    for(int task_begin = 0; task_begin < align_end; task_begin += task_size){
      auto ret = _M_pool.submit(
        [task_begin, task_size, row, offset, &fn, &output, &cargs...]
        {return std::forward<Fn>(fn) 
          (task_begin, task_size, row, offset, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    // rest of tasks
    int task_begin = align_end, task_sz = output_col % task_size;
    auto ret = _M_pool.submit(
      [task_begin, task_sz, row, offset, &fn, &output, &cargs...]
      {return std::forward<Fn>(fn) 
        (task_begin, task_sz, row, offset, output, std::cref(cargs)...);
      });
    rets.push_back(std::move(ret));
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_channel(Fn&& fn, std::shared_ptr<Tensor<Tp>> output, 
  int offset, Ts&&... cargs) {
    int output_ch = output->channel();
    int square = output->row() * output->col();
    int task_size = (output_ch / nthread) ? output_ch / nthread : 1;
    int align_end = output_ch - output_ch % task_size;

    for(int task_begin = 0; task_begin < align_end; task_begin += task_size){
      auto ret = _M_pool.submit(
        [task_begin, task_size, square, offset, &fn, &output, &cargs...]
        {return std::forward<Fn>(fn)
        (task_begin, task_size, square, offset, output, std::cref(cargs)...);
        });
      rets.push_back(std::move(ret));
    }
    // rest of tasks
    int task_begin = align_end, task_sz = output_ch % task_size;
    auto ret = _M_pool.submit(
      [task_begin, task_sz, square, offset, &fn, &output, &cargs...]
      {return std::forward<Fn>(fn) 
        (task_begin, task_sz, square, offset, output, std::cref(cargs)...);
      });
    rets.push_back(std::move(ret));
  }

  template<typename Fn, typename Tp, typename... Ts>
  void parallel_number(Fn&& fn, std::shared_ptr<Tensor<Tp>> output, Ts&&... cargs) {
    int output_num = output->number();
    int volume = output->row() * output->col() * output->channel();
    int task_size = (output_num / nthread) ? output_num / nthread : 1;
    int align_end = output_num - output_num % task_size;

    for(int task_begin = 0; task_begin < align_end; task_begin += task_size){
      auto ret = _M_pool.submit(
        [task_begin, task_size, volume, &fn, &output, &cargs...]
        {return std::forward<Fn>(fn)
        (task_begin, task_size, volume, 0, output, std::cref(cargs)...);
         //task_begin, task_num, shape, offset
        });
      rets.push_back(std::move(ret));
    }
    // rest of tasks
    int task_begin = align_end, task_sz = output_num % task_size;
    auto ret = _M_pool.submit(
      [task_begin, task_sz, volume, &fn, &output, &cargs...]
      {return std::forward<Fn>(fn)
      (task_begin, task_sz, volume, 0, output, std::cref(cargs)...);
      });
    rets.push_back(std::move(ret));
  }

  /*
  this function should be called at every layer's forward()
  to avoid data race.
  */
  void sync(){ 
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