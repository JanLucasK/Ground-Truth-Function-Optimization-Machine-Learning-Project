import pandas as pd
import os
import matplotlib.pyplot as plt

def read_data_from_files(file_names):
    all_data = []

    for file_name in file_names:
        with open(file_name, 'r') as file:
            lines = file.readlines()

        data = {'Function': [], 'Seed': [], 'Neural-Net Optimum': [], 'Ground-Truth Optimum': [], 'Distance': [], 'Optimizer': []}

        current_function = None
        current_seed = None
        current_nn_optimum = None
        current_distance = None
        found_ground_truth_optimum = False

        # Add the optimizer column based on the file name
        if 'pso' in file_name.lower():
            current_optimizer = 'PSO'
        else:
            current_optimizer = 'BasinHopping'

        for line in lines:
            line = line.strip()
            if line.startswith('f_'):
                if current_function is not None:
                    # Remove the seed number from the function (e.g., from "f_01_0" to "f_01")
                    function_name = '_'.join(current_function.split('_')[:-1])
                    data['Function'].append(function_name)
                    data['Seed'].append(current_seed)
                    data['Neural-Net Optimum'].append(current_nn_optimum)
                    if found_ground_truth_optimum:
                        data['Ground-Truth Optimum'].append(current_gt_optimum)
                    else:
                        data['Ground-Truth Optimum'].append(None)
                    data['Distance'].append(current_distance)
                    data['Optimizer'].append(current_optimizer)
                current_function = line
                current_seed = int(line.split('_')[-1])
                found_ground_truth_optimum = False
            elif line.startswith('neural-net optium:'):
                current_nn_optimum = [float(val) for val in line.split('[')[1].split(']')[0].split()]
            elif line.startswith('ground-truth optimum'):
                current_gt_optimum = [float(val) for val in line.split('[')[1].split(']')[0].split()]
                found_ground_truth_optimum = True
            elif line.startswith('distance:'):
                current_distance = float(line.split(':')[-1].strip())

        # Add the last data points
        if current_function is not None:
            # Remove the seed number from the function (e.g., from "f_01_0" to "f_01")
            function_name = '_'.join(current_function.split('_')[:-1])
            data['Function'].append(function_name)
            data['Seed'].append(current_seed)
            data['Neural-Net Optimum'].append(current_nn_optimum)
            if found_ground_truth_optimum:
                data['Ground-Truth Optimum'].append(current_gt_optimum)
            else:
                data['Ground-Truth Optimum'].append(None)
            data['Distance'].append(current_distance)
            data['Optimizer'].append(current_optimizer)

        # Add the data from this file to the overall data list
        all_data.append(data)

    return all_data

def create_combined_dataframe(all_data):
    df = pd.DataFrame(all_data[0])  # Use the first set of data as the base

    # Add data from the second file (if available)
    if len(all_data) > 1:
        df2 = pd.DataFrame(all_data[1])
        df = pd.concat([df, df2], ignore_index=True)

    return df

def plot_scatterplots(df):
    # Create separate scatterplots for each function side by side
    functions = df['Function'].unique()
    num_functions = len(functions)

    max_distance = df['Distance'].max()  # Determine the y-axis limits for all scatterplots

    plt.figure(figsize=(18, 6))

    for i, func in enumerate(functions):
        func_df = df[df['Function'] == func]
        ax = plt.subplot(1, num_functions, i + 1)

        pso_data = func_df[func_df['Optimizer'] == 'PSO']
        basin_data = func_df[func_df['Optimizer'] == 'BasinHopping']

        ax.scatter(pso_data['Seed'], pso_data['Distance'], s=100, marker='o', color='blue', label='PSO')
        ax.scatter(basin_data['Seed'], basin_data['Distance'], s=100, marker='o', color='green', label='BasinHopping')

        ax.set_xlabel('Seed')
        ax.set_ylabel('Distance')
        ax.set_title(f'Function {func}')

        ax.set_ylim(0, max_distance + 0.5)  # Set the y-axis limits
        ax.set_xticks(func_df['Seed'].unique())  # Show only existing seed values on the x-axis

        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig('compare_opt_distance.png')
    plt.show()

if __name__ == "__main__":
    # List of file names
    file_names = ['pso_results_50SwarmSize.txt', 'basin_results.txt']
    
    all_data = read_data_from_files(file_names)
    df = create_combined_dataframe(all_data)
    plot_scatterplots(df)
