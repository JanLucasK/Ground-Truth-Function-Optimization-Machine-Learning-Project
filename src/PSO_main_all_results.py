import csv
from bin.PSO_opt_withoutVisualization import PSO_optimizer  # Import the PSO_optimizer class
import numpy as np
import os

def get_function_from_model_name(model_name):
    if "f01" in model_name:
        return 'f_01'
    elif "f03" in model_name:
        return 'f_03'
    elif "f24" in model_name:
        return 'f_24'
    else:
        return 'Unknown'

def get_version_from_path(path):
    parts = path.split('/')
    if len(parts) >= 2:
        return parts[-2]  # Get the second to last part of the path
    else:
        return 'Unknown'

def main():
    input_bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    swarmsize = 50  # Number of particles each swarm
    niter = 50  # Number of iterations

    # List of subdirectories containing the models
    model_subdirs = ["models/v1", "models/v2", "models/v3"]

    pso_opt = PSO_optimizer(input_bounds=input_bounds)

    # Create a CSV file for results
    with open('opt_results/pso_results_all_50SwarmSize_50niter.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Path', 'Version', 'Function', 'Seed', 'Neural_net_optimum', 'Ground_truth_optimum', 'Distance', 'Optimizer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each subdirectory
        for model_dir in model_subdirs:
            # List of model files in the subdirectory
            model_files = os.listdir(model_dir)

            # Iterate over model files in the subdirectory
            for model_file in model_files:
                if model_file.endswith(".pth"):
                    model_name = os.path.splitext(model_file)[0]
                    seed = 0
                    for _ in range(4):
                        name = f"{model_name}_{seed}"
                        np.random.seed(seed)
                        model_path = os.path.join(model_dir, model_file)
                        result_nn, result_bbob = pso_opt.optimize(model_path=model_path, function=model_name,
                                                                   swarmsize=swarmsize,
                                                                   niter=niter, seed=seed,
                                                                   save_image=True, image_name=name)
                        print(name)
                        print(result_nn)
                        print(result_bbob)

                        # Calculate the distance between the two optimization results using Euclidean norm
                        distance = round(np.linalg.norm(result_nn - result_bbob), 5)
                        print(distance)

                        # Get the function from the model name
                        function = get_function_from_model_name(model_name)

                        # Get the version from the model path
                        version = get_version_from_path(model_path)

                        # Write the results to the CSV file
                        writer.writerow({'Path': model_path,
                                         'Version': version,
                                         'Function': function,
                                         'Seed': seed,
                                         'Neural_net_optimum': result_nn,
                                         'Ground_truth_optimum': result_bbob,
                                         'Distance': distance,
                                         'Optimizer': 'PSO'})
                        seed += 1

if __name__ == "__main__":
    main()
