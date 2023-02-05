#pragma once

// #define BENCH
#ifdef BENCH
#include <basic/timer.hh>
#endif

#include <future>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <exception>

#include <iostream>

namespace dl{

static std::vector<std::future<int>> _M_pool;
static std::mutex _M_mtx;

inline static size_t 
cpu_number() noexcept {
  return std::thread::hardware_concurrency();
}

static void 
clean_task(){
  size_t size = _M_pool.size();
  std::cout << "size:" << size << std::endl;
  if(size == 0) return;

  std::unique_lock lck{_M_mtx};
  for(auto& end = *(_M_pool.end() - 1); auto& task : _M_pool){
    try{
      printf("In loop");
      auto ret = task.get();
      std::cout << ret << " \n"[&end  == &task];
    } 
    catch(std::exception& e){
      std::cout << "[exception caught:]\n";
    }
  } 
  _M_pool.clear();
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
  printf("ch_mod:%d\n", ch_mod);
  for(int i = 0; i < nthread; i++){
    try{
      auto fut = std::async(std::launch::deferred, 
      /*thread function   */std::forward<Fn>(f),
      /*ch_begin, ch_num  */ch_num * i, ch_num, 0,
      /*res, const args...*/std::ref(res), std::cref(cargs)...);
      std::cout << "Ready to push_back\n";
      _M_pool.push_back(std::move(fut));
    }
    catch(std::exception& e){
      std::cout << "Got exception\n";
    }
  }      //       ch_begin,    ch_num, pad, res, args...
  if(ch_mod) f(channel - ch_mod, ch_mod, 0, res, cargs...);

  clean_task();
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
  for(int ch = 0; ch < channel; ch++){
    for(int i = 0; i < nthread; i++){
      auto fut = std::async(std::launch::async, 
      /*thread function   */std::forward<Fn>(f),
      /*row_be, row_n, ch */row_num * i, row_num, ch,
      /*res, const args...*/std::ref(res), std::cref(cargs)...);
      _M_pool.push_back(std::move(fut));
    }         //     row_begin,  row_num, channel, res, args...
    if(row_mod) f(row - row_mod, row_mod, ch, res, cargs...);
  }
  clean_task();
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
  for(int ch = 0; ch < channel; ch++){
    for(int i = 0; i < nthread; i++){
      auto fut = std::async(std::launch::async, 
      /*thread function   */std::forward<Fn>(f),
      /*col_be, col_n, ch */col_num * i, col_num, ch,
      /*res, const args...*/std::ref(res), std::cref(cargs)...);
      _M_pool.push_back(std::move(fut));
    }         //     col_begin,  col_num, channel, res, args...
    if(col_mod) f(col - col_mod, col_mod, ch, res, cargs...);
  }
  clean_task();
}

} // namespace dl