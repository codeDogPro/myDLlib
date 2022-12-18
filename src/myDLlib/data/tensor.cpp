#include <data/tensor.h>
#include <basic/enumaration.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <future>

#include <numeric>
#include <algorithm>

#include <memory>

#include <assert.h>
#include <iostream>

namespace dl{

  template<typename T>
  static void 
  vec_cal // the row number of a must >= b
  (const Tensor<T> &a, const Tensor<T> &b, Tensor<T> *result,
   int row_begin, int row_num, int col, int mode){
    for(int i = 0; i < row_num; i++){
      int row_idx = (row_begin + i) * col;
      for(int j = 0; j < col; j++){
        switch(mode){
          case PLUS:
          result[row_idx + col] = a[row_idx + col] + b[col]; break;
          case MINUS:
          result[row_idx + col] = a[row_idx + col] - b[col]; break;
          case MULTIPLY:
          result[row_idx + col] = a[row_idx + col] * b[col]; break;
          case DIVIDE:
          result[row_idx + col] = a[row_idx + col] / b[col]; break;
          default: assert(0);
        }
      }
    }
  }


  template<typename T>
  Tensor<T> *
  calculator(const Tensor<T> &a, const Tensor<T> &b, int mode){
    assert(a.m_shape[1] == b.m_shape[1]);

    auto result = new Tensor<T>(a.m_shape);

    if(a.m_shape[0] != b.m_shape[0]){
      if(b.m_shape[0] == 1){
        // TODO:mode b to each row of a
        int ncpu = std::thread::hardware_concurrency();
        int row_num = a.m_shape[0] / ncpu;
        printf("ncpu: %d\n", ncpu);

        std::vector<std::thread> pool;
        for(int i = 0; i < ncpu; i++){
          int row_begin = i * row_num;
          std::thread task(vec_cal, a, b, result, row_begin,
                           row_num, a.m_shape[1], mode);
          pool.push_back(std::move(task));
        }
        for(auto &task : pool) task.join();

        return result;
      }

      fprintf(stderr,
      "The size of tensor a:(%d) must match the size of tensor b:(%d) \
      at non-singleton dimension 0\n", a.m_shape[0], b.m_shape[0]);
      assert(0);
    }
    else{
      return result;
      // TODO:invoke matrix mode()
    }
  }

}