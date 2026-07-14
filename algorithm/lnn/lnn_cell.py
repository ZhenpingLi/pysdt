import numpy as np
from typing import List

class LNNCell:
    """
    A Liquid Time-Constant (LTC) recurrent neural network cell, implemented using NumPy.
    This model uses a system of ordinary differential equations (ODEs) to model
    continuous-time dynamics.
    """

    def __init__(self, input_size: int, hidden_size: int, time_step: float):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_step = time_step
        
        # --- All attributes are now NumPy arrays ---
        
        # Weights for the combined input (current input + previous state)
        # Initialized with random weights for better starting point (Glorot/Xavier initialization)
        self.weights = np.random.randn(hidden_size, input_size + hidden_size) * np.sqrt(1.0 / (input_size + hidden_size))
        self.bias = np.zeros(hidden_size)
        
        # The hidden state of the cell
        self.state = np.zeros(hidden_size)
        
        # Liquid Time-Constant (LTC) parameters
        # Time constant for each neuron in the hidden state
        self.time_constants = np.ones(hidden_size) # Initialize with a default value of 1.0
            
        self.last_combined_input: np.ndarray = None

    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the LNN cell.
        
        :param input_vec: A 1D NumPy array representing the current input.
        :return: The updated hidden state of the cell as a 1D NumPy array.
        """
        # 1. Combine the current input with the previous hidden state
        combined = np.concatenate((input_vec, self.state))
        self.last_combined_input = combined

        # 2. Compute the linear part of the state update, followed by activation
        # linear_part = W * combined + b
        linear_part = np.dot(self.weights, combined) + self.bias
        activated_part = np.tanh(linear_part)

        # 3. Solve the ODE for the state update using Forward Euler
        # dx/dt = -x/tau + f(input, state)
        # where f is the activated linear part
        d_state_dt = (-self.state / self.time_constants) + activated_part
        
        # 4. Update the state using the derivative and time step
        # new_state = old_state + (d_state/dt) * dt
        self.state += d_state_dt * self.time_step
        
        return self.state

    # --- Getters and Setters for NumPy arrays ---
    def get_weights(self) -> np.ndarray:
        return self.weights

    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    def get_bias(self) -> np.ndarray:
        return self.bias

    def set_bias(self, bias: np.ndarray):
        self.bias = bias

    def get_state(self) -> np.ndarray:
        return self.state

    def get_last_combined_input(self) -> np.ndarray:
        return self.last_combined_input
