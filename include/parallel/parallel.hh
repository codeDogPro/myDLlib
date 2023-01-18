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

static void 
clean_task(auto& pool){
  std::cout << "size:" << pool.size() << std::endl;
  for(auto& end = *(pool.end() - 1); auto& task : pool){
    auto ret = task.get();
    std::cout << ret << " \n"[&end  == &task];
  } 
}

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
    /*thread function   */std::forward<Fn>(f),
    /*ch_begin, ch_num  */ch_num * i, ch_num,
    /*res, const args...*/std::ref(res), std::cref(cargs)...);
    pool.push_back(std::move(fut));
  }      //       ch_begin,      ch_num, res, args...
  if(ch_mod) f(channel - ch_mod, ch_mod, res, cargs...);

  clean_task(pool);
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
      /*thread function   */std::forward<Fn>(f),
      /*row_be, row_n, ch */row_num * i, row_num, ch,
      /*res, const args...*/std::ref(res), std::cref(cargs)...);
      pool.push_back(std::move(fut));
    }         //     row_begin,  row_num, channel, res, args...
    if(row_mod) f(row - row_mod, row_mod, ch, res, cargs...);
  }
  clean_task(pool);
}

template<typename Fn, typename Tp, typename... Ts>
void
parallel_col(Fn&& f, int nthread, Tensor<Tp>& res, Ts&&... cargs){
#ifdef BENCH
    Timer t;
#endif
  puts("In parallel_col");
  int col = res.col(), channel = res.channel();
  int col_num = col / nthread, col_mod = col % nthread;
  std::vector<std::future<int>> pool;
  for(int ch = 0; ch < channel; ch++){
    for(int i = 0; i < nthread; i++){
      auto fut = std::async(std::launch::async, 
      /*thread function   */std::forward<Fn>(f),
      /*col_be, col_n, ch */col_num * i, col_num, ch,
      /*res, const args...*/std::ref(res), std::cref(cargs)...);
      pool.push_back(std::move(fut));
    }         //     col_begin,  col_num, channel, res, args...
    if(col_mod) f(col - col_mod, col_mod, ch, res, cargs...);
  }
  clean_task(pool);
}

} // namespace dl