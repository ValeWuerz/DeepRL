#include <iostream>
#include <fstream>
#include "/usr/include/nlohmann/json.hpp"
#include "torch/torch.h"

using namespace::std;
using json = nlohmann::json;

int main() {
   
    ifstream file("blob_data.json");
    vector<float> vec = {1.0243, -12.2731, 3.0, 4.0, 5.0};
    json data;

    file >> data;
 
    cout << data << endl;
  
  // create vector from JSON data
    std::vector<float> data_vec;
    for (const auto& row : data["data"]) {
        for (const auto& val : row) {
            data_vec.push_back(val.get<float>());
    }
  }
auto t = torch::from_blob(data_vec.data(),{2,40}, torch::kFloat32);

  // print tensor
cout << "Tensor: " << t << std::endl;

  return 0; 

}
