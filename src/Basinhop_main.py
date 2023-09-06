from bin.Basinhopping_optimizer import Basinhopping_optimizer as bh_optimizer
import sys
import numpy as np
from RMSE_Calc import rmse_calc

#naming-schema: function _ size _ seed

def main():
    start = [0,0]
    step_size = 1
    temp=100
    n_iter = 100
    models = [
        ["f_01", "models/v3/training_v3_f01_5.pth", "f_01_5"],
       #["f_03","models/v3/training_v3_f03_5.pth", "f_03_5"],
        ["f_24", "models/v3/training_v3_f24_5.pth","f_24_5"],
       #["f_01", "models/v3/training_v3_f01_1.pth", "f_01_1"],
       # ["f_01", "models/v1/training_v1_f01_1.pth", "f_01_1"],
        #["f_03","models/v3/training_v3_f03_1.pth", "f_03_1"],
        ["f_24", "models/v3/training_v3_f24_1.pth", "f_24_1"],
        #["f_01", "models/v3/training_v3_f01_3.pth", "f_01_3"],
        ["f_24","models/training_v4_f24_4.pth", "f_24_4"],
        ["f_03", "models/training_v5_f03_4.pth", "f_03_5"]
        ]
    
    basehop_opt = bh_optimizer(input_bounds=[(-5.0,5.0), (-5.0,5.0)])
    for model in models:
        seed = 0
        for _ in range(4):
            #name = model[0]+"_"+str(seed)
            name = f"{model[2]}_{seed}" 
            np.random.seed(seed)
            result_nn, result_bbob, nn_path, bbob_path = basehop_opt.optimize(model_path= model[1], function =model[0], initial_guess = start, niter = n_iter,
                                                               stepsize=step_size, T=temp, seed=seed,
                                                               save_image=True, image_name=name)
            print(model[2]+"_"+str(seed))
            print(result_nn.x)
            print(result_bbob.x)
            distance = np.linalg.norm(result_nn.x-result_bbob.x)
            print(distance)
            
            calculator = rmse_calc(x_y_coordinates= nn_path, model_path=model[1], function_name=model[0])
            rmse =calculator.evaluate_model_and_bbob()
            print(rmse)
            
            with open('opt_results/basin_results.txt', "a") as f:
                f.write(f"{name}")
                f.write(f"\n")
                f.write(f"neural-net optium: {result_nn.x}, ground-thruth optimum: {result_bbob.x}")
                f.write(f"\n")
                f.write(f"distance: {distance}")
                f.write(f"\n")
                f.write(f"rmse: {rmse}")
                f.write(f"\n")
            seed+=1
if __name__ == "__main__":
    main()