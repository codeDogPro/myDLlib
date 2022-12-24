#pragma once

#include <string>
#include <cstdlib>
#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

namespace dl{

template<typename T>
std::string type_name(){
  const char *name = typeid(T).name();
#if defined(__GNUC__) || defined(__clang__)
  int status;
  char *p = abi::__cxa_demangle(name, 0, 0, &status);
  std::string s = p;
  free(p);
#else
  std::string s = name;
#endif
  if(std::is_const_v<std::remove_reference_t<T>>)
    s += " const";
  if(std::is_volatile_v<std::remove_reference_t<T>>)
    s += " volatile";
  if(std::is_lvalue_reference_v<T>)
    s += " &";
  if(std::is_rvalue_reference_v<T>)
    s += " &&";
  return s;
}

}
