import json
import torch.nn as nn
import torch
from sklearn.datasets import make_blobs
import numpy

data = [1,2,3]
x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
x_train = torch.FloatTensor(x_train)
print(x_train)
tensor_dict = {'data': x_train.tolist(), 'dtype': str(x_train.dtype), 'device': str(x_train.device)}
print(tensor_dict)
with open('build/blob_data.json', 'w') as outfile:
    json.dump(tensor_dict, outfile)
#with open("data.json", "w") as f:
    # Serialize the dictionary to a JSON string and write it to the file
#    json.dump(data, f)