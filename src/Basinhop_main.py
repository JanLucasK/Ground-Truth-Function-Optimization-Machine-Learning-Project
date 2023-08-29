from bin.Basinhopping_optimizer import Basinhopping_optimizer as bh_optimizer
import sys
import numpy as np
def main():
    start = [0,0]
    step_size = 0.5
    temp=100
    models = [
        ["f_01", "models/v3/training_v3_f01_4.pth"],
        ["f_03","models/training_v4_f03_5.pth"],
        ["f_24", "models/training_v4_f24_4.pth"]
        ]
    basehop_opt = bh_optimizer(input_bounds=[(-5.0,5.0), (-5.0,5.0)])
    for model in models:
        seed = 0
        for _ in range(3):
            name = model[0]+"_"+str(seed)
            result_nn, result_bbob, fig = basehop_opt.optimize(model_path= model[1], function =model[0], initial_guess = start, 
                                                               stepsize=step_size, T=temp, seed= seed, 
                                                               save_image=True, image_name=name)
            print(model[0]+"_"+str(seed))
            print(result_nn.x)
            print(result_bbob.x)
            #distance = basehop_opt.calc_distance(point_a=result_nn.x, point_b=result_bbob.x)
            distance = np.linalg.norm(result_nn.x-result_bbob.x)
            print(distance)
            with open('basin_results.txt', "a") as f:
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