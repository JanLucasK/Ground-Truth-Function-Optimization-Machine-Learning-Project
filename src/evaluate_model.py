from RMSE_Calc import rmse_calc
import os
import numpy as np
import torch
import warnings
import pandas as pd


# Disable all Python warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")


folder_path = 'models/v6/'  # Replace with the actual path to your folder

# Get a list of all filenames in the folder
filenames = os.listdir(folder_path)


# Generate the grid of input data
x_grid, y_grid = np.meshgrid(np.arange(-5, 5.01, 0.05), np.arange(-5, 5.01, 0.05))

grid_data = np.column_stack((x_grid.ravel(), y_grid.ravel()))


results_rmse = []

# Print the list of filenames
for filename in filenames:
    split_filename = filename.split("_")

    Calc = rmse_calc(grid_data, folder_path+filename, split_filename[2].replace("f","f_"))
    rmse = Calc.evaluate_model_and_bbob()

    results_rmse.append(rmse)
    print(rmse)

df = pd.DataFrame(data={"model":filenames,"rmse":results_rmse})
df.to_csv("data/processed/rmse_models.csv", index=None)