import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import bbobtorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bin.SimpleNeuralNet import SimpleNeuralNet
from torch.utils.data import DataLoader, TensorDataset


class Trainer():
    def __init__(self, data_file_path, seed, neurons_per_layer, num_layers, learning_rate, batch_size, num_epochs, save_image, image_name, save_model, model_name, gt_function_show):

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available and being used")
        else:
            device = torch.device("cpu")
            print("GPU is not available, using CPU instead")


        self.data_file_path = data_file_path
        self.seed = seed
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_image = save_image
        self.image_name = image_name
        self.save_model = save_model
        self.model_name = model_name
        self.gt_function_show = gt_function_show

        self._load_data()
        # self._scale_data()
        self._data_to_tensors()
        self._create_dataloader()
        self._initialize_model()


    def _load_data(self):

        data = pd.read_parquet(self.data_file_path)

        # Extracting features (x and y) and labels (k)
        X = data[['coord_0', 'coord_1']].values
        y = data['f_value'].values
        

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)


    def _scale_data(self):
        # Standardize the features (optional but generally recommended)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def _data_to_tensors(self):

        # Convert NumPy arrays to PyTorch tensors
        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

    def _create_dataloader(self):
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def _initialize_model(self):
        self.model = SimpleNeuralNet(self.neurons_per_layer, self.num_layers)
        # Loss function
        self.criterion = nn.MSELoss()
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def _get_gt_function(self, x_grid, y_grid):
        function_name = self.data_file_path.split("/")[-1][0:4]
        
        if function_name == 'f_01':
            fn = bbobtorch.create_f01(2, seed=42)  # two dimension with seed 42
        elif function_name == 'f_03':
            fn = bbobtorch.create_f03(2, seed=42)  # two dimension with seed 42
        elif function_name == 'f_24':
            fn = bbobtorch.create_f24(2, seed=42)  # two dimension with seed 42

        flat_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Convert flat_grid to a PyTorch Tensor
        flat_grid_tensor = torch.tensor(flat_grid, dtype=torch.float32)

        # Evaluate the ground-truth function using the bbobtorch function
        results = fn(flat_grid_tensor)

        return results.numpy().reshape(x_grid.shape)

    def _plot_heatmap(self, x_grid, y_grid, predictions, function_values, gt_function_show):
        
        if gt_function_show:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            sns.heatmap(predictions, cmap='viridis', xticklabels=False, yticklabels=False, ax=axes[0])
            axes[0].set_xlabel('X Coordinate')
            axes[0].set_ylabel('Y Coordinate')
            axes[0].set_title('Heatmap of Model Predictions')

            sns.heatmap(function_values, cmap='viridis', xticklabels=False, yticklabels=False, ax=axes[1])
            axes[1].set_xlabel('X Coordinate')
            axes[1].set_ylabel('Y Coordinate')
            axes[1].set_title('Heatmap of BBOB Function')

            plt.tight_layout()
        
        else:

            sns.heatmap(predictions, cmap='viridis', xticklabels=False, yticklabels=False)
            plt.tight_layout()
            

        if self.save_image:
            # Save the plot as a PNG file
            function_name = self.data_file_path.split("/")[-1][0:4]
            plt.savefig(f'images/{self.image_name}', dpi=300)  # You can adjust the dpi (dots per inch) as needed
            plt.clf()



    def train(self):
        for epoch in range(self.num_epochs):
            for X_batch, y_batch in self.train_loader:
                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(X_batch)
                # Compute the loss
                loss = self.criterion(outputs.view(-1), y_batch)
                # Backward pass
                loss.backward()
                # Update the weights
                self.optimizer.step()


            # Evaluate the model on the test set
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(self.X_test_tensor)
                test_loss = self.criterion(test_outputs.view(-1), self.y_test_tensor)

            print(f"Epoch {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}")

        self._save_model(self.model, self.model_name)

    def evaluate_grid(self):
        # Generate the grid of input data
        x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.01), np.arange(-5, 5.01, 0.01))

        # Scale the grid data using the same StandardScaler
        # grid_data_scaled = self.scaler.transform(np.column_stack((x_grid.ravel(), y_grid.ravel())))
        grid_data = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Convert the scaled grid data to PyTorch tensor
        grid_data_tensor = torch.tensor(grid_data, dtype=torch.float32)

        # Set the model to evaluation mode
        self.model.eval()

        # Pass the grid data through the model to get predictions
        with torch.no_grad():
            predictions = self.model(grid_data_tensor).view(x_grid.shape)

        # Get the ground-truth function values on the grid
        function_values = self._get_gt_function(x_grid, y_grid)

        # Plot the heatmaps
        self._plot_heatmap(x_grid, y_grid, predictions, function_values, self.gt_function_show)

    def _save_model(self, model, file_name):
        if self.save_model:
            torch.save(model, f'models/{file_name}')

