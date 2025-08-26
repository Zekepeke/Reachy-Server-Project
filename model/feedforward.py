import numpy as np
import torch
import torch.nn as nn

class Feedforward(nn.Module):

    def __init__(self, input_size, num_classes):

      super().__init__()

      # nn.Sequential is a container that allows to build neural networks
      # in a sequential, layer-by-layer format.
      self.ff = nn.Sequential (
          # I'm guessing the number of inputs of labels?
          #

          # CHANGED nn.Linear(input_size, 20) TO nn.Linear(input_size, 100) FOR NO MISMATCH
          nn.Linear(input_size, 100), # Input layer with the max length of the feature list, intermediate with 20 nodes
          nn.ReLU(), # Activation function
          nn.Linear(100, 100), # Intermediate layer with 100 nodes
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, 100),
          nn.ReLU(),
          nn.Linear(100, num_classes), # Output layer with `num_classes` nodes
          nn.Softmax(dim=1)  # Softmax for probabilities
      )

    def forward(self, x):
        return self.ff(x)
    
class LandmarkPointsClassifier(object):
    def __init__(
        self,
        length=42,
        num_classes=4,
        model_path="",
        score_th=0.4,
        invalid_value=0,
        num_threads=1,
    ):
        # Initialize the Feedforward model with the correct input size and number of classes
        self.model = Feedforward(input_size=length, num_classes=num_classes)
        
        # self.model.load_state_dict(state_dict)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # Adjust for nested structure
        if "model" in state_dict:
            state_dict = state_dict["model"]

        self.model.load_state_dict(state_dict)


        # Set the model to evaluation mode
        self.model.eval()
        
        
        self.score_th = score_th
        self.invalid_value = invalid_value
        self.num_threads = num_threads
        
        # Set number of threads for cpu based inference
        torch.set_num_threads(self.num_threads)
    
    def __call__(self, point_history_cords: list[float]):
        # Convert input data to a PyTorch tensor and add batch dimension
        input_tensor = torch.tensor([point_history_cords], dtype=torch.float32)

        # Perform forward pass
        with torch.no_grad():  # Disables gradient calculation for efficiency
            output = self.model(input_tensor)
        
        # Apply softmax to get probabilities and find the max probability and class
        probabilities = torch.softmax(output, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        print("For landmark max prob: ", max_prob.item())
        print("For landmark predicted class: ", predicted_class.item())
        print("For landmark all predictions: ", probabilities)
        
        # Check if the maximum probability meets the score threshold
        if max_prob.item() >= self.score_th:
            return predicted_class.item()
        else:
            print("invalid value " + str(self.invalid_value))
            return self.invalid_value
