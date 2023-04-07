#include <torch/torch.h>

class MyNet : public torch::nn::Module {
public:
  MyNet(int input_size, int hidden_size, int output_size)
      : fc1(input_size, hidden_size),
        fc2(hidden_size, output_size) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = torch::sigmoid(fc2(x));
    return x;
  }

private:
  torch::nn::Linear fc1, fc2;
};

int main() {
  const int input_size = 10;
  const int hidden_size = 5;
  const int output_size = 1;

  MyNet net(input_size, hidden_size, output_size);

  torch::Tensor input = torch::randn({1, input_size});
  torch::Tensor target = torch::randn({1, output_size});

  // define loss and optimizer
  torch::nn::BCELoss criterion;
  torch::optim::SGD optimizer(net.parameters(), /*lr=*/0.01);

  for (int i = 0; i < 100; ++i) {
    // zero gradients
    optimizer.zero_grad();

    // forward pass
    //torch::Tensor output = net.forward(input);
    auto output2= net.forward(input); 
    // compute loss
    torch::Tensor loss = criterion(output2, target);

  std::cout << "Input: " << loss << std::endl;
    // backward pass
    loss.backward();

    // update parameters
    optimizer.step();
  }

  std::cout << "Input: " << input << std::endl;
  std::cout << "Output: " << net.forward(input) << std::endl;

  return 0;
}
