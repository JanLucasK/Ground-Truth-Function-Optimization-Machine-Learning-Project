import torch
import torch.nn as nn

class SimpleNeuralNet(nn.Module):
    def __init__(self, neurons_per_layer, num_layers):
        super(SimpleNeuralNet, self).__init__()

        # Initialize the neural network with the specified number of neurons per layer and the number of layers.
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        
        # Initialize a list to store the layers of the neural network.
        self.fc = []
        
        # Add the input layer with 2 input features and 'neurons_per_layer' hidden units (32 in this case).
        self.fc.append(nn.Linear(2, self.neurons_per_layer)) 

        # Add 'num_layers' hidden layers with 'neurons_per_layer' units each.
        for i in range(0, self.num_layers):
            self.fc.append(nn.Linear(self.neurons_per_layer, self.neurons_per_layer))
        
        # Add the output layer with 'neurons_per_layer' hidden units and 1 output unit.
        self.fc.append(nn.Linear(self.neurons_per_layer, 1))

        # Create a ModuleList to manage the layers.
        self.fc = nn.ModuleList(self.fc)
        
    # Define the forward method to specify how data flows through the network.
    def forward(self, x):
        for layer in self.fc[:-1]:
            # Apply a ReLU activation function to each hidden layer's output.
            x = torch.relu(layer(x))
        
        # Pass the final layer's output through without activation.
        x = self.fc[-1](x)
        
        return x
