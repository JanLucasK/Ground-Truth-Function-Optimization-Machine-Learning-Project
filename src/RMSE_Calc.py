import numpy as np
import torch
import bbobtorch
from sklearn.metrics import mean_squared_error


class rmse_calc:

    def __init__(self, x_y_coordinates, model_path, function_name, input_bounds=[-5,5]):
        self.input_bounds = input_bounds
        self.x_y_coordinates = x_y_coordinates
        self.model = self.load_model(model_path)
        self.function_name = function_name

    def load_model(self, path):
       return torch.load(path)
   
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
            for val in coords:
                if val < self.input_bounds[0] or val > self.input_bounds[1]:
                    prediction = torch.tensor(0, dtype=torch.float32)
            # Append the prediction to the predictions array
            predictions.append(prediction.item())


            # Calculate the BBOB value based on the function name
            if self.function_name == 'f_01':
                fn = bbobtorch.create_f01(2, seed=42)  # two dimension with seed 42
            elif self.function_name == 'f_03':
                fn = bbobtorch.create_f03(2, seed=42)  # two dimension with seed 42
            elif self.function_name == 'f_24':
                fn = bbobtorch.create_f24(2, seed=42)  # two dimension with seed 42
            
            bbob_value = fn(torch.tensor(np.column_stack((coords[0], coords[1])), dtype=torch.float32))
            for val in coords:
                if val < self.input_bounds[0] or val > self.input_bounds[1]:
                    bbob_value = 0
            bbob_values.append(bbob_value)

        #print(predictions)
        #print(bbob_values)
        
        
        # Calculate RMSE between predictions and BBOB values
        rmse = np.sqrt(mean_squared_error(predictions, bbob_values))
        print(rmse)
        return rmse


#test_array = [[3.0,4.0],[5.0,4.0],[-1.0,-2.0],[3.0,3.0],[4.0,-5.0],[4.0,3.0],[1.0,1.0],[2.0,1.0],[4.0,2.0],[4.0,3.0]]
#test_model = ("models/v3/training_v3_f01_3.pth")
#test_function_string = "f_01"

<<<<<<< HEAD
#calculator = rmse_calc(test_array,test_model,test_function_string)
#print(calculator.evaluate_model_and_bbob())
=======
calculator = rmse_calc(test_array,test_model,test_function_string,[-5,5])
print(calculator.evaluate_model_and_bbob())
>>>>>>> 017f4cbc54fb59a127254ec67c854e368b0d100f
