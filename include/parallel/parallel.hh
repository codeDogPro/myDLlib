#pragma once

// #define BENCH
#ifdef BENCH
#include <basic/timer.hh>
#endif

#include <parallel/tensor_calcu.hh>

#include <future>
#include <vector>

#include <iostream>

namespace dl{

template<typename Fn, typename Tp, typename... Ts>
void 
parallel_channel(Fn&& f, int nthread, Tensor<Tp>& res, Ts&&... cargs) {
#ifdef BENCH
    Timer t;
#endif
  puts("In parallel_channel");
  int channel = res.channel();
  int ch_num = channel / nthread, ch_mod = channel % nthread;
  std::vector<std::future<int>> pool;
  for(int i = 0; i < nthread; i++){
    auto fut = std::async(std::launch::async, 
                          std::forward<Fn>(f),
                          ch_num * i, ch_num,
                          std::ref(res), std::cref(cargs)...);
    pool.push_back(std::move(fut));
  }
  if(ch_mod) f(channel - ch_mod, ch_mod, res, cargs...);

  for(auto& task : pool){
    auto ret = task.get();
    std::cout << ret << ' ';
  } 
}

template<typename Fn, typename Tp, typename... Ts>
void 
parallel_row(Fn&& f, int nthread, Tensor<Tp>& res, Ts&&... cargs){
#ifdef BENCH
    Timer t;
#endif
  puts("In parallel_row");
  int row = res.row(), channel = res.channel();
  int row_num = row / nthread, row_mod = row % nthread;
  std::vector<std::future<int>> pool;
  for(int ch = 0; ch < channel; ch++){
    for(int i = 0; i < nthread; i++){
      auto fut = std::async(std::launch::async, 
                          std::forward<Fn>(f),
                          row_num * i, row_num, ch,
                          std::ref(res), std::cref(cargs)...);
      pool.push_back(std::move(fut));
    }         //     row_begin,  row_num, channel, res, args...
    if(row_mod) f(channel - row_mod, row_mod, ch, res, cargs...);
  }
  for(auto& task : pool){
    auto ret = task.get();
    std::cout << ret << ' ';
  }
}

template<typename Fn, typename Tp, typename... Ts>
void
parallel_col(Fn&& f, int nthread, Tensor<Tp>& res, Ts&&... args){

}

} // namespace dl