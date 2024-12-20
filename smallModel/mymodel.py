import torch
import torch.nn as nn
import torch.nn.functional as F

#documentation for neural network: https://pytorch.org/docs/stable/nn.html#convolution-layers

numInputs = 21
numOutputs = 1

layer1Nodes = 64 #4 is too small
layer2Nodes = 64 #5 is also too sm all
layer3Nodes = 32
layer4Nodes = 32

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ###other possible layers
        '''self.L1 = nn.Linear(7, 10)
        self.reLu = nn.ReLU()
        self.L2 = nn.Linear(10, 4)'''

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(numInputs, layer1Nodes), #add a dropout?
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(layer1Nodes, layer2Nodes),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            nn.Linear(layer2Nodes, numOutputs)
            #nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            #nn.Linear(layer3Nodes, layer4Nodes),
            #nn.LeakyReLU(),
            #nn.Dropout(p=0.2),
            #nn.Linear(layer4Nodes, numOutputs)
        )

    def forward(self, x):
        #super easy forward propogation
        '''x = x
        z1 = self.L1(x)
        a1 = self.reLu(z1)
        z2 = self.L2(a1)

        return z2'''

        logits = self.linear_relu_stack(x)
        return logits