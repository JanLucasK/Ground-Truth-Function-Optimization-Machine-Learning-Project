import csv
from bin.Basinhopping_optimizer import Basinhopping_optimizer as bh_optimizer
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
    
def get_size_from_path(path):
    if "_1" in path:
        return '100'
    elif "_2" in path:
        return '1000'
    elif "_3" in path:
        return '10000'
    elif "_4" in path:
        return '100000'
    elif "_5" in path:
        return '1000000'
    else:
        return 'Unknown'

def main():
    input_bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    start = [0,0]
    step_size = 0.5
    temp=100
    n_iter = 100

    # List of subdirectories containing the models
    model_subdirs = ["models/v1",
                     "models/v2", 
                     "models/v3"
                     ]

    basehop_opt = bh_optimizer(input_bounds=[(-5.0,5.0), (-5.0,5.0)])

    # Create a CSV file for results
    with open('opt_results/basehop_results_all.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Path', 'Version', 'Function', 'Size', 'Seed', 'Neural_net_optimum', 'Ground_truth_optimum', 'Distance', 'Optimizer']
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
                        result_nn, result_bbob = basehop_opt.optimize(model_path=model_path, function =model_name, initial_guess = start, niter = n_iter,
                                                               stepsize=step_size, T=temp, seed= seed, 
                                                               save_image=False, image_name=name)
                        #result_nn = result_nn.x
                        #result_bbob = result_bbob.x
                        print(name)
                        print(result_nn)
                        print(result_bbob)

                        # Calculate the distance between the two optimization results using Euclidean norm
                        distance = round(np.linalg.norm(result_nn.x- result_bbob.x), 5)
                        print(distance)

                        # Get the function from the model name
                        function = get_function_from_model_name(model_name)
                        
                        size = get_size_from_path(model_path)
                        
                        # Get the version from the model path
                        version = get_version_from_path(model_path)

                        # Write the results to the CSV file
                        writer.writerow({'Path': model_path,
                                         'Version': version,
                                         'Function': function,
                                         'Size': size,
                                         'Seed': seed,
                                         'Neural_net_optimum': result_nn.x,
                                         'Ground_truth_optimum': result_bbob.x,
                                         'Distance': distance,
                                         'Optimizer': 'basinhop'})
                        seed += 1

if __name__ == "__main__":
    main()
