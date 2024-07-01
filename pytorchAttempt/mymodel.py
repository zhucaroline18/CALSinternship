import torch
import torch.nn as nn
import torch.nn.functional as F

#documentation for neural network: https://pytorch.org/docs/stable/nn.html#convolution-layers

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ###other possible layers
        self.L1 = nn.Linear(7, 10)
        self.reLu = nn.ReLU()
        self.L2 = nn.Linear(10, 4)

        '''self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 10), 
            nn.ReLU(),
            nn.Linear(10, 4)
        )'''


    def forward(self, x):
        #super easy forward propogation
        x = x
        z1 = self.L1(x)
        a1 = self.reLu(z1)
        z2 = self.L2(a1)

        return z2

        '''logits = self.linear_relu_stack(x)
        return logits'''