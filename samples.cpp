#include <torch/torch.h>
#include <vector>
#include <iostream>

int main() {
  int testing = 0;
  torch::Tensor tensor = torch::rand({5, 5});
  std::cout << tensor + 1 << std::endl;
  testing = 1;
}