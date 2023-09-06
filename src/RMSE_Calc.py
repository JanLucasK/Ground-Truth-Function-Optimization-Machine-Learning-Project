import numpy as np
import torch
import bbobtorch
from sklearn.metrics import mean_squared_error


class RSME_Calc:

    def __init__(self, x_y_coordinates, model, function_name):

        self.x_y_coordinates = x_y_coordinates
        self.model = model
        self.function_name = function_name

    def evaluate_model_and_bbob(self):
        # Load the PyTorch model
        self.model.eval()
        
        # Initialize arrays to store predictions and BBOB values
        predictions = []
        bbob_values = []
        
        # Loop through each set of coordinates
        for coords in self.x_y_coordinates:
            # Convert coordinates to PyTorch tensor
            inputs = torch.tensor(coords, dtype=torch.float32)
            
            # Generate predictions from the model
            with torch.no_grad():
                prediction = self.model(inputs)
            
            # Append the prediction to the predictions array
            predictions.append(prediction.item())


            # Calculate the BBOB value based on the function name
            if self.function_name == 'f_01':
                fn = bbobtorch.create_f01(2, seed=42)  # two dimension with seed 42
            elif self.function_name == 'f_03':
                fn = bbobtorch.create_f03(2, seed=42)  # two dimension with seed 42
            elif self.function_name == 'f_24':
                fn = bbobtorch.create_f24(2, seed=42)  # two dimension with seed 42
    
            bbob_values.append(fn(torch.tensor(np.column_stack((coords[0], coords[1])), dtype=torch.float32)))

        print(predictions)
        print(bbob_values)
        
        
        # Calculate RMSE between predictions and BBOB values
        rmse = np.sqrt(mean_squared_error(predictions, bbob_values))
        
        return rmse


test_array = [[3.0,4.0],[5.0,4.0],[-1.0,-2.0],[3.0,3.0],[4.0,-5.0],[4.0,3.0],[1.0,1.0],[2.0,1.0],[4.0,2.0],[4.0,3.0]]
test_model = torch.load("models/v3/training_v3_f01_3.pth")
test_function_string = "f_01"

calculator = RSME_Calc(test_array,test_model,test_function_string)
print(calculator.evaluate_model_and_bbob())
