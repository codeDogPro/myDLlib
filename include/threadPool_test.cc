#include <vector>
#include <thread>
#include <boost/circular_buffer.hpp>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

class task {
public:
  task() = default;

  template <class Func>
  explicit task(Func &&f) : ptr_{new wrapper{std::move(f)}} {}

  void operator()() { ptr_->call(); }

private:
  class wrapper_base {
  public:
    virtual void call() = 0;
    virtual ~wrapper_base() = default;
  };

  template <class Func>
  class wrapper : public wrapper_base {
  public:
    explicit wrapper(Func &&f) : f_{std::move(f)} {}
    virtual void call() override { f_(); }

  private:
    Func f_;
  };

  std::unique_ptr<wrapper_base> ptr_;
};

class thread_pool {
public:
  explicit thread_pool(size_t n) : q_{n}, running_{false} {}

  ~thread_pool() { stop(); }

  void start(size_t n) {
    if (running_) return;

    running_ = true;
    threads_.reserve(n);
    while (n--)
      threads_.emplace_back(&thread_pool::worker, this);
  }

  void stop() {
    if (!running_) return;
    {
      std::lock_guard lk{m_};
      running_ = false;
      not_full_.notify_all();
      not_empty_.notify_all();
    }
    for (auto &t : threads_) {
      if (t.joinable())
        t.join();
    }
  }

  template <class Fun, class Ret = std::invoke_result_t<Fun>>
  std::future<Ret> 
  submit(Fun f) {
    std::packaged_task<Ret()> pt{std::move(f)};
    auto ret = pt.get_future();
    task t{std::move(pt)};
    {
      std::unique_lock lk{m_};
      not_full_.wait(lk, [this]
                     { return !running_ || !q_.full(); });
      assert(!running_ || !q_.full());

      if (!running_) return {};

      q_.push_back(std::move(t));
      not_empty_.notify_one();
    }
    return ret;
  }

private:
  void worker() {
    while (true) {
      task t;
      {
        std::unique_lock lk{m_};
        not_empty_.wait(lk, [this]
                        { return !running_ || !q_.empty(); });

        assert(!running_ || !q_.empty());

        if (!running_)
          return;

        t = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
      }
      t();
    }
  }

  std::vector<std::thread> threads_;
  boost::circular_buffer<task> q_;
  std::mutex m_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  bool running_;
};

#include <iostream>

int test(std::vector<int> &v, bool b, int ret){
  if(b){
    v.assign(5, 1);
    for(int i = 0; auto x : v) printf("v[%d]: %d\n", i++, x);
  }
  return ret;
}

thread_pool pool{10};

template<typename Fn, typename Tp, typename... Ts>
void submit(Fn &&f, std::vector<Tp> &v, Ts&&... args){
  auto ret = pool.submit([&]
              {return std::forward<Fn>(f)(v, std::ref(args)...);});
  std::cout << ret.get() << std::endl;
}

class Tensor{
public:
  Tensor(int n, int num) : _data(n, num) {}
  const std::vector<int> &
  data(){
    return _data;
  }
private:
  std::vector<int> _data;
};

int main() {
  // pool.start(3);
  // std::vector<int> v;
  // for(int i = 0; i < 10; i++){
  //   submit(test, v, i % 2 == 0, i);
  // }
  std::shared_ptr<Tensor> ptr = std::make_shared<Tensor>(5, 2);
  size_t n = ptr->data().size();
  for(int i = 0; i < n; i++){
    std::cout << ptr->data()[i] << ' ';
  }
  std::cout << std::endl;
  // std::vector<std::future<int>> rets;
  // for (int i = 0; i < 100; i++) {
  //   std::vector<int> v;
  //   auto ret = pool.submit([&]
  //                          { return test(v, true, i); });
  //   rets.push_back(std::move(ret));
  // }
  // for (auto &ret : rets) {
  //   std::cout << ret.get() << " ";
  // }
}
