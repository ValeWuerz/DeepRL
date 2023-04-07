#include <iostream>
#include <torch/torch.h>
using namespace std;
using namespace torch;

class Net : public nn::Module{
public:
Net(int input_size, int hidden_size, int output_size):
fc1(input_size, hidden_size),
relu_act(),
fc2(hidden_size, output_size),
sigmoid_act()
{
 register_module("fc1", fc1);
 register_module("relu", relu_act);
 register_module("fc2", fc2);
 register_module("sigmoid", sigmoid_act);
}
Tensor forward(Tensor x_train){
  //x_train = torch::relu(fc1(x_train));
  //x_train = torch::sigmoid(fc2(x_train));
  Tensor hidden = fc1(x_train);
  Tensor relu_ = relu_act(hidden); 
  Tensor hidden2 = fc2(relu_);
  Tensor output = sigmoid_act(hidden2);
  return output;
}
private:
  nn::Linear fc1{nullptr}, fc2{nullptr};
  nn::ReLU relu_act{nullptr};
  nn::Sigmoid sigmoid_act{nullptr};
};

int main() {
  const int num_samples = 40;
  const int hidden_size = 10;
  const int num_features = 2;
  const int input_size = num_features;
  const int output_size=1;
  const int learning_rate=0.1;
  const int epochs=100;
  
  Tensor x_train = torch::rand({num_samples, num_features});
  Tensor y_train = torch::empty({num_samples, 1}).uniform_(0,2).to(torch::kInt).to(torch::kFloat);

  Tensor x_test = torch::rand({10, 2});
  Tensor y_test = torch::empty({10, 1}).uniform_(0,2).to(torch::kInt).to(torch::kFloat);
  

  Net my_net(input_size, hidden_size, output_size);

  nn::BCELoss criterion;
  optim::SGD optimizer(my_net.parameters(), learning_rate);

  for (int i = 0;i< epochs; ++i){
    //zero gradients
    optimizer.zero_grad();

    // forward pass
  Tensor output = my_net.forward(x_train);

  //  compute loss
  Tensor loss = criterion(output, y_train);

  cout << loss.item() << endl; 
  //  backward pass
  loss.backward();

  // update parameters
  optimizer.step();


  }
} 