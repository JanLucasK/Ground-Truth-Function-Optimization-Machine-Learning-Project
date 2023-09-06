import csv
from bin.PSO_opt_withoutVisualization import PSO_optimizer  # Import the PSO_optimizer class
import numpy as np
from RMSE_Calc import rmse_calc

def main():
    input_bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    swarmsize = 50  # Number of particles in each swarm
    niter = 20  # Number of iterations

    # List of model files to iterate over
    model_files = [
        ["f_01", "models/v1/training_v1_f01_1.pth"],
        ["f_01", "models/v1/training_v1_f01_2.pth"],
        ["f_01", "models/v1/training_v1_f01_3.pth"],
        ["f_03", "models/v1/training_v1_f03_1.pth"],
        ["f_03", "models/v1/training_v1_f03_2.pth"],
        ["f_03", "models/v1/training_v1_f03_3.pth"],
        ["f_24", "models/v1/training_v1_f24_1.pth"],
        ["f_24", "models/v1/training_v1_f24_2.pth"],
        ["f_24", "models/v1/training_v1_f24_3.pth"],
        ["f_01", "models/v2/training_v2_f01_1.pth"],
        ["f_01", "models/v2/training_v2_f01_2.pth"],
        ["f_01", "models/v2/training_v2_f01_3.pth"],
        ["f_03", "models/v2/training_v2_f03_1.pth"],
        ["f_03", "models/v2/training_v2_f03_2.pth"],
        ["f_03", "models/v2/training_v2_f03_3.pth"],
        ["f_24", "models/v2/training_v2_f24_1.pth"],
        ["f_24", "models/v2/training_v2_f24_2.pth"],
        ["f_24", "models/v2/training_v2_f24_3.pth"],
        ["f_01", "models/v3/training_v3_f01_1.pth"],
        ["f_01", "models/v3/training_v3_f01_2.pth"],
        ["f_01", "models/v3/training_v3_f01_3.pth"],
        ["f_01", "models/v3/training_v3_f01_4.pth"],
        ["f_01", "models/v3/training_v3_f01_5.pth"],
        ["f_03", "models/v3/training_v3_f03_1.pth"],
        ["f_03", "models/v3/training_v3_f03_2.pth"],
        ["f_03", "models/v3/training_v3_f03_3.pth"],
        ["f_03", "models/v3/training_v3_f03_4.pth"],
        ["f_03", "models/v3/training_v3_f03_5.pth"],
        ["f_24", "models/v3/training_v3_f24_1.pth"],
        ["f_24", "models/v3/training_v3_f24_2.pth"],
        ["f_24", "models/v3/training_v3_f24_3.pth"],
        ["f_24", "models/v3/training_v3_f24_4.pth"],
        ["f_24", "models/v3/training_v3_f24_5.pth"]
    ]
    
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
        
    pso_opt = PSO_optimizer(input_bounds=input_bounds)

    # Create a CSV file for results
    with open('opt_results/pso_results_all_50SS_20niter.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Path', 'Version', 'Function', 'Size', 'Seed', 'Neural_net_optimum', 'Ground_truth_optimum', 'Distance', 'RMSE', 'Optimizer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for model in model_files:
            model_name = model[0]
            model_path = model[1]
            seed = 0
            for _ in range(20):
                name = f"{model_name}_{seed}"
                np.random.seed(seed)
                result_nn, result_bbob, nn_path_df = pso_opt.optimize(model_path=model_path, function=model_name,
                                                               swarmsize=swarmsize,
                                                               niter=niter, seed=seed,
                                                               save_image=True, image_name=name)
                print(name)
                print(result_nn)
                print(result_bbob)

                # Calculate the distance between the two optimization results using Euclidean norm
                distance = round(np.linalg.norm(result_nn - result_bbob), 5)
                print(distance)
                
                nn_path = nn_path_df[['x1', 'x2']].to_numpy()
                # Calculate RMSE
                calculator = rmse_calc(x_y_coordinates= nn_path, model_path=model_path, function_name=model_name)
                rmse =calculator.evaluate_model_and_bbob()
                print(rmse)
                
                # Get the size from the model path
                size = get_size_from_path(model_path)

                # Get the version from the model path
                version = get_version_from_path(model_path)

                # Write the results to the CSV file
                writer.writerow({'Path': model_path,
                                 'Version': version,
                                 'Function': model_name,
                                 'Size': size,
                                 'Seed': seed,
                                 'Neural_net_optimum': result_nn,
                                 'Ground_truth_optimum': result_bbob,
                                 'Distance': distance,
                                 'RMSE': rmse,
                                 'Optimizer': 'PSO'})
                seed += 1

if __name__ == "__main__":
    main()
