#pragma once

#include <basic/function.cuh>

namespace dl{

/*
                           RxCxCHxN
Flatten make tensor shape: 1x1x2048x1 --> 1x2048x1x1,
which to match the Linear layer.
*/
template<typename T=f32>
class Flatten : public Function<T> {
public:
  Flatten(int _start_dim=1, int _end_dim=-1){
    start_dim = _start_dim;
    if(_end_dim == -1){
      end_dim = 3;
    }
    else end_dim = _end_dim;
  } 

  virtual std::shared_ptr<Tensor<T>>
  forward(const std::shared_ptr<Tensor<T>> input) override{
    int irow = input->row(), icol = input->col();
    int ichannel = input->channel(), inumber = input->number();
    int orow, ocol, ochannel, onumber;
    if(start_dim == 0 && end_dim == 3){
      ocol = inumber * ichannel * irow * icol;
      onumber = ochannel = orow = 1;
    }
    else if(start_dim == 0 && end_dim == 2){
      orow = inumber * ichannel * irow;
      ocol = icol;
      onumber = ochannel = 1;
    }
    else if(start_dim == 0 && end_dim == 1){
      ochannel = inumber * ichannel;
      orow = irow;
      ocol = icol;
      onumber = 1;
    }
    else if(start_dim == 1 && end_dim == 3){
      orow = inumber;
      ocol = ichannel * irow * icol;
      onumber = ochannel = 1;
    }
    else if(start_dim == 1 && end_dim == 2){
      orow = ichannel * irow; 
      ocol = icol;
      ochannel = inumber;
      onumber = 1;
    }
    else if(start_dim == 2 && end_dim == 3){
      ocol = irow * icol;
      orow = ichannel;
      ochannel = inumber;
      onumber = 1;
    }
    else if(start_dim == end_dim){
      orow = irow, ocol = icol, ochannel = ichannel;
      onumber = inumber;
    }
    auto output = std::make_shared<Tensor<T>>(input->get_cshape(), Device::CPU, 0);
    *output = *input;
    output->reshape(orow, ocol, ochannel, onumber);
    return output;
  }

private:
  int start_dim, end_dim;
};
}