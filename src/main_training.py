import sys
import os
import json
from bin.Training_SimpleNeuralNet import Trainer
from bin.SimpleNeuralNet import SimpleNeuralNet
from bin.Basinhopping_optimizer import Basinhopping_optimizer as bh_optimizer


def open_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config

def main():

    # Load a configuration from a JSON file 
    config = open_config("config/training_version_6.json")

    # Iterate over the range of data files specified in the configuration.
    for i in range(0, len(config["data_file_path"])):
        
        # Create an instance of the Trainer class with various configuration parameters from the loaded JSON.
        trainer = Trainer(
            config["data_file_path"][i], 
            config["seed"][i], 
            config["neurons_per_layer"][i], 
            config["num_layers"][i], 
            config["learning_rate"][i], 
            config["batch_size"][i], 
            config["num_epochs"][i], 
            config["save_image"][i], 
            config["image_name"][i],
            config["save_model"][i],
            config["model_name"][i],
            config["gt_function_show"][i]
        )
        
        # Train the neural network using the specified configuration.
        trainer.train()
        
        # Evaluate the trained neural network using a grid.
        trainer.evaluate_grid()

# Check if the script is being run as the main program.
# If it is, execute the main() function.
if __name__ == "__main__":
    main()