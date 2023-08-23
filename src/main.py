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

    config = open_config("config/training_version_1.json")
    for i in range(0,len(config["data_file_path"])):
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
        )
        trainer.train()
        trainer.evaluate_grid()     
   
if __name__ == "__main__":
    main()