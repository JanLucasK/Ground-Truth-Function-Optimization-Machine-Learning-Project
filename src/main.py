import sys
import os
from bin.Training_SimpleNeuralNet import Trainer


def main():

    trainer = Trainer()

    trainer.train()
    trainer.evaluate_grid()

if __name__ == "__main__":
    main()