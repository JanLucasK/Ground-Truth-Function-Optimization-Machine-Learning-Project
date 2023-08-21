import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class visualization():
    def __init__(self, model, function) -> None:
        self.model = model
        self.function = function
        self.scaler = StandardScaler()

    def _get_gt_function(self, x_grid, y_grid):
        fn = function
        flat_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Convert flat_grid to a PyTorch Tensor
        flat_grid_tensor = torch.tensor(flat_grid, dtype=torch.float32)

        # Evaluate the ground-truth function using the bbobtorch function
        results = fn(flat_grid_tensor)

        return results.numpy().reshape(x_grid.shape) 
    
    def evaluate_grid(self, nn_results, bbob_results):
        # Generate the grid of input data
        x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.01), np.arange(-5, 5.01, 0.01))


        # Convert the scaled grid data to PyTorch tensor
        grid_data_tensor = torch.tensor((x_grid, y_grid), dtype=torch.float32)

        # Set the model to evaluation mode
        self.model.eval()

        # Pass the grid data through the model to get predictions
        with torch.no_grad():
            predictions = self.model(grid_data_tensor).view(x_grid.shape)

        # Get the ground-truth function values on the grid
        function_values = self._get_gt_function(x_grid, y_grid)

        # Plot the heatmaps
        self._plot_heatmap(x_grid, y_grid, predictions, function_values, nn_results, bbob_results)
    
    def plot_heatmap(self, x_grid, y_grid, predictions, function_values, nn_results, bbob_results):
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


        # Overlay datapoints on the heatmap
        
        plt.text(bbob_results.x[0], bbob_results.x[1], str(bbob_results.fun), ha='center', va='center', color='red', weight='bold')

        plt.show() 