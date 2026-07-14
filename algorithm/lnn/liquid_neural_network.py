import numpy as np
from typing import List, Sequence

# Assuming lnn_cell is in the same directory
from algorithm.lnn.lnn_cell import LNNCell

class LiquidNeuralNetwork:
    """
    A Liquid Neural Network (LNN) model, which is a type of continuous-time
    recurrent neural network. This implementation uses a Liquid Time-Constant (LTC)
    cell as its core component.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, time_step: float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.cell = LNNCell(input_size, hidden_size, time_step)
        
        # Output layer weights and biases
        self.output_weights = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.output_bias = np.zeros(output_size)

    def forward_sequence(self, input_sequence: Sequence[np.ndarray]) -> List[np.ndarray]:
        """
        Processes a sequence of inputs through the network.

        :param input_sequence: A list or sequence of 1D NumPy arrays, where each array
                               is an input at a single time step.
        :return: A list of 1D NumPy arrays representing the output at each time step.
        """
        outputs = []
        # Reset the cell's state before processing a new sequence
        self.cell.state = np.zeros(self.hidden_size)
        
        for input_vec in input_sequence:
            output = self.forward_step(input_vec)
            outputs.append(output)
            
        return outputs

    def forward_step(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Performs a single forward pass (one time step) through the network.
        """
        # 1. Get the next hidden state from the LTC cell
        hidden_state = self.cell.forward(input_vec)
        
        # 2. Calculate the output layer's linear combination
        # output = W_out * hidden_state + b_out
        output = np.dot(self.output_weights, hidden_state) + self.output_bias
        
        # 3. Apply the final activation function (sigmoid)
        # The sigmoid function is applied element-wise
        sigmoid_output = 1 / (1 + np.exp(-output))
        
        return sigmoid_output

    def get_weights_flat(self) -> np.ndarray:
        """
        Flattens all weights and biases of the network into a single 1D array.
        """
        # Use np.concatenate for efficient joining of flattened arrays
        return np.concatenate([
            self.cell.weights.flatten(),
            self.cell.bias.flatten(),
            self.output_weights.flatten(),
            self.output_bias.flatten()
        ])

    def set_weights_flat(self, all_weights: np.ndarray):
        """
        Sets all weights and biases of the network from a single flat 1D array.
        """
        index = 0
        
        # Un-flatten cell weights
        cell_w_size = self.cell.weights.size
        self.cell.weights = all_weights[index : index + cell_w_size].reshape(self.cell.weights.shape)
        index += cell_w_size
        
        # Un-flatten cell bias
        cell_b_size = self.cell.bias.size
        self.cell.bias = all_weights[index : index + cell_b_size]
        index += cell_b_size
        
        # Un-flatten output weights
        output_w_size = self.output_weights.size
        self.output_weights = all_weights[index : index + output_w_size].reshape(self.output_weights.shape)
        index += output_w_size
        
        # Un-flatten output bias
        output_b_size = self.output_bias.size
        self.output_bias = all_weights[index : index + output_b_size]
        
    @property
    def lnn_dim(self) -> int:
        """
        Calculates the total number of trainable parameters in the network.
        """
        return self.cell.weights.size + self.cell.bias.size + self.output_weights.size + self.output_bias.size
