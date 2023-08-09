import torch
import torch.nn as nn


class SimpleNeuralNet(nn.Module):
    def __init__(self, hidden_layers):
        super(SimpleNeuralNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(2, self.hidden_layers)  # 2 input features, 32 hidden units
        self.fc2 = nn.Linear(self.hidden_layers,self.hidden_layers)
        self.fc3 = nn.Linear(self.hidden_layers,self.hidden_layers)
        self.fc4 = nn.Linear(self.hidden_layers,self.hidden_layers)
        self.fc5 = nn.Linear(self.hidden_layers, 1)  # 32 hidden units, 1 output

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
