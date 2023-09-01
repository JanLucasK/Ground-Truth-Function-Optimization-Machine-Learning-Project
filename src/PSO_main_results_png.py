from bin.PSO_optimizer import PSO_optimizer  # Import the PSO_optimizer class
import numpy as np

def main():
    input_bounds=[(-5.0,5.0), (-5.0,5.0)]
    swarmsize = 50  # Number of particles each swarm
    niter=20 # Number of iterations

    # List of optimization tasks with model paths
    models = [
        ["f_01", "models/v3/training_v3_f01_4.pth"],
        ["f_03","models/v3/training_v3_f03_5.pth"],
        ["f_24", "models/v3/training_v3_f24_4.pth"]
        ]
    pso_opt = PSO_optimizer(input_bounds=input_bounds)
    
    # Iterate over different optimization tasks
    for model in models:
        seed = 0
        for _ in range(3):
            name = model[0]+"_"+str(seed)
            result_nn, result_bbob, fig = pso_opt.optimize(model_path= model[1], function =model[0],  
                                                               swarmsize=swarmsize, 
                                                               niter=niter, seed= seed, 
                                                               save_image=True, image_name=name)
            print(model[0]+"_"+str(seed))
            print(result_nn)
            print(result_bbob)
            
            # Calculate the distance between the two optimization results using Euclidean norm
            distance = np.linalg.norm(result_nn-result_bbob) 
            print(distance)
            
            # Write the task name to the results file
            with open('pso_results_50SwarmSize.txt', "a") as f:
                f.write(f"{name}")
                f.write(f"\n")
                f.write(f"neural-net optium: {result_nn}, ground-thruth optimum: {result_bbob}")
                f.write(f"\n")
                f.write(f"distance: {distance}")
                f.write(f"\n")
                f.write(f"\n")
            seed+=1

if __name__ == "__main__":
    main()
    