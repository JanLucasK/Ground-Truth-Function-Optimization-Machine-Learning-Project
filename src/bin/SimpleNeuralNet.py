import torch
import torch.nn as nn


class SimpleNeuralNet(nn.Module):
    def __init__(self, neurons_per_layer, num_layers):
        super(SimpleNeuralNet, self).__init__()

        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.fc = []
        self.fc.append(nn.Linear(2, self.neurons_per_layer))  # 2 input features, 32 hidden units
        for i in range(0,self.num_layers):
            self.fc.append(nn.Linear(self.neurons_per_layer, self.neurons_per_layer))
        self.fc.append(nn.Linear(self.neurons_per_layer, 1))  # 32 hidden units, 1 output

        self.fc = nn.ModuleList(self.fc)
        
    def forward(self, x):
        for layer in self.fc[:-1]:
            x = torch.relu(layer(x))
        x = self.fc[-1](x)
        return x
