import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import bbobtorch

# Set the device to CPU
device = torch.device('cpu')

x = torch.arange(-5,5, 0.001, dtype=torch.float32)
grid = torch.stack(torch.meshgrid(x, x), -1)
flat_grid = torch.reshape(grid, (-1,2))
xgrid, ygrid = np.meshgrid(x.numpy(), x.numpy())


functions = ['01','03','24']
samples = [100,1000,10000,100000,1000000]

for x_function in functions:

    if x_function == '01':
        fn = bbobtorch.create_f01(2, seed=42)  # two dimension with seed 42
    elif x_function == '03':
        fn = bbobtorch.create_f03(2, seed=42)  # two dimension with seed 42
    elif x_function == '24':
        fn = bbobtorch.create_f24(2, seed=42)  # two dimension with seed 42


    results = fn(flat_grid)
    results_grid = torch.reshape(results, xgrid.shape) - fn.f_opt

    # Convert flat_grid and results to numpy arrays for easier handling
    flat_grid_np = flat_grid.numpy()
    results_np = results.numpy()

    # Create a DataFrame
    df = pd.DataFrame(flat_grid_np, columns=[f'coord_{i}' for i in range(flat_grid_np.shape[1])])
    df['f_value'] = results_np

    for sample in samples:
        random_rows = df.sample(n=sample, replace=False)

        # Save the DataFrame to a CSV file
        random_rows.to_parquet(f'./data/raw/f_{x_function}_s_{sample}.parquet', index=False)
