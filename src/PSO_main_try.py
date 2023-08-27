from bin.PSO_optimizer import PSO_optimizer  # Import the PSO_optimizer class
import bbobtorch

def main():
    
    # Define the BBOB function you want to use
    bbob_function = bbobtorch.create_f01(2, seed=42)  # Replace with the desired BBOB function

    # Create an instance of the PSO_optimizer class and run the optimization
    model_path = 'models/training_v1_f01_3.pth'
    input_bounds = input_bounds=[(-5.0, 5.0), (-5.0, 5.0)]
    niter = 20  # Number of iterations
    num_particles = 50  # Number of particles each swarm

    optimizer = PSO_optimizer(model_path, input_bounds, bbob_function, num_particles)  # Pass the BBOB function
    initial_guess = [0.0, 0.0]
    
    result_nn, result_bbob, fig = optimizer.optimize(initial_guess, niter)

    # Adjust the path where the animation GIF will be saved
    anim_filename = "/home/luka/Documents/Test_01_pso_animation.gif"
    fig.savefig(anim_filename)  # Save the final visualization

if __name__ == "__main__":
    main()
