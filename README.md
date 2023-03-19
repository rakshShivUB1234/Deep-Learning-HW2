# Deep-Learning-HW2
DL hw2
I have coded in python and used PYCHARM software. To run the program: I have used the packages called 
"pip install pytorch"


import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


**the output is:**

/Users/rakshithashivaraj/Downloads/FNN_mnist/bin/python /Users/rakshithashivaraj/Documents/PycharmCodes/FNN_mnist/dlHW2.py 
Epoch [1/10], Step [100/600], Training Loss: 1.6805
Epoch [1/10], Step [200/600], Training Loss: 1.6041
Epoch [1/10], Step [300/600], Training Loss: 1.5915
Epoch [1/10], Step [400/600], Training Loss: 1.5480
Epoch [1/10], Step [500/600], Training Loss: 1.5406
Epoch [1/10], Step [600/600], Training Loss: 1.5543
Epoch [1/10], Test Accuracy: 92.81%
Epoch [2/10], Step [100/600], Training Loss: 1.5659
Epoch [2/10], Step [200/600], Training Loss: 1.5555
Epoch [2/10], Step [300/600], Training Loss: 1.5309
Epoch [2/10], Step [400/600], Training Loss: 1.5080
Epoch [2/10], Step [500/600], Training Loss: 1.5639
Epoch [2/10], Step [600/600], Training Loss: 1.5826
Epoch [2/10], Test Accuracy: 93.08%
Epoch [3/10], Step [100/600], Training Loss: 1.5211
Epoch [3/10], Step [200/600], Training Loss: 1.5048
Epoch [3/10], Step [300/600], Training Loss: 1.5137
Epoch [3/10], Step [400/600], Training Loss: 1.5041
Epoch [3/10], Step [500/600], Training Loss: 1.5140
Epoch [3/10], Step [600/600], Training Loss: 1.5044
Epoch [3/10], Test Accuracy: 95.05%
Epoch [4/10], Step [100/600], Training Loss: 1.4816
Epoch [4/10], Step [200/600], Training Loss: 1.4820
Epoch [4/10], Step [300/600], Training Loss: 1.4836
Epoch [4/10], Step [400/600], Training Loss: 1.5049
Epoch [4/10], Step [500/600], Training Loss: 1.4834
Epoch [4/10], Step [600/600], Training Loss: 1.5353
Epoch [4/10], Test Accuracy: 95.71%
Epoch [5/10], Step [100/600], Training Loss: 1.4969
Epoch [5/10], Step [200/600], Training Loss: 1.5100
Epoch [5/10], Step [300/600], Training Loss: 1.4836
Epoch [5/10], Step [400/600], Training Loss: 1.4966
Epoch [5/10], Step [500/600], Training Loss: 1.5092
Epoch [5/10], Step [600/600], Training Loss: 1.5097
Epoch [5/10], Test Accuracy: 96.10%
Epoch [6/10], Step [100/600], Training Loss: 1.5158
Epoch [6/10], Step [200/600], Training Loss: 1.4879
Epoch [6/10], Step [300/600], Training Loss: 1.5022
Epoch [6/10], Step [400/600], Training Loss: 1.4926
Epoch [6/10], Step [500/600], Training Loss: 1.5221
Epoch [6/10], Step [600/600], Training Loss: 1.4953
Epoch [6/10], Test Accuracy: 96.25%
Epoch [7/10], Step [100/600], Training Loss: 1.4850
Epoch [7/10], Step [200/600], Training Loss: 1.5130
Epoch [7/10], Step [300/600], Training Loss: 1.5005
Epoch [7/10], Step [400/600], Training Loss: 1.4972
Epoch [7/10], Step [500/600], Training Loss: 1.4747
Epoch [7/10], Step [600/600], Training Loss: 1.4951
Epoch [7/10], Test Accuracy: 96.53%
Epoch [8/10], Step [100/600], Training Loss: 1.5011
Epoch [8/10], Step [200/600], Training Loss: 1.4631
Epoch [8/10], Step [300/600], Training Loss: 1.5236
Epoch [8/10], Step [400/600], Training Loss: 1.4812
Epoch [8/10], Step [500/600], Training Loss: 1.4833
Epoch [8/10], Step [600/600], Training Loss: 1.4799
Epoch [8/10], Test Accuracy: 96.52%
Epoch [9/10], Step [100/600], Training Loss: 1.4634
Epoch [9/10], Step [200/600], Training Loss: 1.4800
Epoch [9/10], Step [300/600], Training Loss: 1.4817
Epoch [9/10], Step [400/600], Training Loss: 1.4823
Epoch [9/10], Step [500/600], Training Loss: 1.5276
Epoch [9/10], Step [600/600], Training Loss: 1.4866
Epoch [9/10], Test Accuracy: 96.52%
Epoch [10/10], Step [100/600], Training Loss: 1.4616
Epoch [10/10], Step [200/600], Training Loss: 1.4926
Epoch [10/10], Step [300/600], Training Loss: 1.4788
Epoch [10/10], Step [400/600], Training Loss: 1.4615
Epoch [10/10], Step [500/600], Training Loss: 1.4744
Epoch [10/10], Step [600/600], Training Loss: 1.4821
Epoch [10/10], Test Accuracy: 96.98%

Process finished with exit code 0
