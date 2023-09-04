from bin.Basinhopping_optimizer import Basinhopping_optimizer as bh_optimizer
import sys
import numpy as np


#naming-schema: function _ size _ seed

def main():
    start = [0,0]
    step_size = 0.5
    temp=100
    n_iter = 100
    models = [
        ["f_01", "models/v3/training_v3_f01_5.pth", "f_01_5"],
        ["f_03","models/v3/training_v3_f03_5.pth", "f_03_5"],
        ["f_24", "models/v3/training_v3_f24_5.pth","f_24_5"],
        ["f_01", "models/v3/training_v3_f01_1.pth", "f_01_1"],
        ["f_03","models/v3/training_v3_f03_1.pth", "f_03_1"],
        ["f_24", "models/v3/training_v3_f24_1.pth", "f_24_1"],
        ["f_01", "models/v3/training_v3_f01_3.pth", "f_01_3"],
        ["f_03","models/v3/training_v3_f03_3.pth", "f_03_3"],
        ["f_24", "models/v3/training_v3_f24_3.pth", "f_24_3"]
        ]
    
    basehop_opt = bh_optimizer(input_bounds=[(-5.0,5.0), (-5.0,5.0)])
    for model in models:
        seed = 0
        for _ in range(4):
            #name = model[0]+"_"+str(seed)
            name = f"{model[2]}_{seed}"  
            result_nn, result_bbob = basehop_opt.optimize(model_path= model[1], function =model[0], initial_guess = start, niter = n_iter,
                                                               stepsize=step_size, T=temp, seed= seed, 
                                                               save_image=True, image_name=name)
            print(model[2]+"_"+str(seed))
            print(result_nn.x)
            print(result_bbob.x)
            distance = np.linalg.norm(result_nn.x-result_bbob.x)
            print(distance)
            
            
            with open('opt_results/basin_results.txt', "a") as f:
                f.write(f"{name}")
                f.write(f"\n")
                f.write(f"neural-net optium: {result_nn.x}, ground-thruth optimum: {result_bbob.x}")
                f.write(f"\n")
                f.write(f"distance: {distance}")
                f.write(f"\n")
                f.write(f"\n")
            seed+=1
if __name__ == "__main__":
    main()