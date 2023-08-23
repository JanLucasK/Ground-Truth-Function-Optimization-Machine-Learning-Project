from bin.Basinhopping_optimizer import Basinhopping_optimizer as bh_optimizer
import sys
def main():
    basehop_opt = bh_optimizer(model_path="models/training_v1_f24_3.pth", input_bounds=[(-5.0,5.0), (-5.0,5.0)])
    result_nn, result_bbob, fig = basehop_opt.optimize(initial_guess = [4,4], stepsize=1, T=200)
    print(result_nn)
    print(result_bbob)
    fig.show()
    
if __name__ == "__main__":
    main()