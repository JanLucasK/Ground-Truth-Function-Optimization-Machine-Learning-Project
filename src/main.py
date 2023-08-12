import sys
import os
import json
from bin.Training_SimpleNeuralNet import Trainer


def open_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config

def main():

    config_file_path = 'config/training_config.json'
    config = open_config(config_file_path)

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
        )
        trainer.train()
        trainer.evaluate_grid()

if __name__ == "__main__":
    main()