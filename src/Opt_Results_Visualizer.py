import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
    plt.savefig('images/compare_opt_Results/compare_opt_distance_v3_5.png')
    plt.show()


def plot_boxplot(csv_1, csv_2):
    # CSV-Dateien einlesen
    csv_file1 = csv_1 # Ersetze 'datei1.csv' durch den Pfad zur ersten CSV-Datei
    csv_file2 = csv_2  # Ersetze 'datei2.csv' durch den Pfad zur zweiten CSV-Datei

    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    # Daten zusammenführen
    combined_df = pd.concat([df1, df2])
    
    max_distance = combined_df['Distance'].max()

    # Boxplots erstellen
    functions = combined_df['Function'].unique()


    for function in functions:
        fig, ax = plt.subplots(figsize=(12,8))
        spacing=1.5
        width= 0.5
        
        #plt.figure(figsize=(10, 6))
    
        # Filtern der Daten für die aktuelle Funktion und PSO bzw. basinhop
        data_pso = combined_df[(combined_df['Function'] == function) & (combined_df['Optimizer'] == 'PSO')]
        data_basinhop = combined_df[(combined_df['Function'] == function) & (combined_df['Optimizer'] == 'basinhop')]
        
        unique_sizes = np.sort(np.unique(np.concatenate([data_pso['Size'].unique(), data_basinhop['Size'].unique()])))

        # Create boxplots
        pso_distances_grouped = [data_pso[data_pso['Size'] == size]['Distance'].values for size in unique_sizes]
        basinhop_distances_grouped = [data_basinhop[data_basinhop['Size'] == size]['Distance'].values for size in unique_sizes]
        
        #plt.boxplot(pso_distances_grouped)
        #plt.boxplot(basinhop_distances_grouped)
        n = len(basinhop_distances_grouped)
        

        positions1 = np.arange(1, 2*n+1, 2) - width/1.5
        box1 = plt.boxplot(pso_distances_grouped, positions=positions1, widths=width, patch_artist=True, boxprops=dict(facecolor='blue', alpha=0.5), labels=[f'List 1 - {i+1}' for i in range(n)])

        # Create boxplots for list2
        positions2 = np.arange(1, 2*n+1, 2) + width/1.5
        box2 = plt.boxplot(basinhop_distances_grouped, positions=positions2, widths=width, patch_artist=True, boxprops=dict(facecolor='green', alpha=0.5), labels=[f'List 2 - {i+1}' for i in range(n)])

        # Label the plot
        plt.title(f'Boxplot für Funktion {function}')    
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.xticks(np.arange(1, 2*n+1, 2), [f'data size {i}' for i in unique_sizes])

        # Add a legend
        ax.legend([box1["boxes"][0], box2["boxes"][0]], ['pso', 'basin hop'], loc='upper right')

        
        
        #plt.ylim([0,max_distance])
        
        #plt.xlabel('Size')
        #plt.ylabel('Distance')
        #plt.xticks(data_pso['Size'].unique())
        #plt.legend(['PSO', 'basinhop'])
        
        #plt.grid(True)
        
        plt.savefig(f'images/compare_opt_Results/boxplot_{function}.png')  # Speichern der Grafik für jede Funktion
        plt.show()


if __name__ == "__main__":
    # List of file names
    #file_names = ['opt_results/pso_results_50SwarmSize_v3_5.txt', 'opt_results/basin_results.txt']
    
    #all_data = read_data_from_files(file_names)
    #df = create_combined_dataframe(all_data)
    #plot_scatterplots(df)
    files = ['opt_results/pso_results_all_50SwarmSize_20niter.csv', 'opt_results/basehop_results_all.csv']
    plot_boxplot(files[0], files[1])


