import numpy as np
from typing import List, Sequence

# Assuming liquid_neural_network is in the same directory
from algorithm.lnn.liquid_neural_network import LiquidNeuralNetwork

class LNNTrainer:
    """
    Handles the training of a LiquidNeuralNetwork using backpropagation through time (BPTT).
    """

    def __init__(self, lnn: LiquidNeuralNetwork, initial_learning_rate: float, decay_rate: float = 0.0, momentum: float = 0.9):
        self.lnn = lnn
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.iteration = 0
        
        # Initialize momentum velocities for each parameter
        self.v_cell_weights = np.zeros_like(lnn.cell.weights)
        self.v_cell_bias = np.zeros_like(lnn.cell.bias)
        self.v_output_weights = np.zeros_like(lnn.output_weights)
        self.v_output_bias = np.zeros_like(lnn.output_bias)

    def train_sequence(self, input_sequence: Sequence[np.ndarray], target_sequence: Sequence[np.ndarray]):
        """
        Trains the network on a full sequence of inputs and targets.
        """
        # Forward pass to get all outputs and hidden states
        outputs = []
        hidden_states = []
        self.lnn.cell.state = np.zeros(self.lnn.hidden_size) # Reset state
        
        for input_vec in input_sequence:
            hidden_state = self.lnn.cell.forward(input_vec)
            hidden_states.append(hidden_state.copy())
            
            # Calculate final output from this hidden state
            output = np.dot(self.lnn.output_weights, hidden_state) + self.lnn.output_bias
            sigmoid_output = 1 / (1 + np.exp(-output))
            outputs.append(sigmoid_output)

        # --- Backward Pass (Backpropagation Through Time) ---
        
        # Initialize gradients to zero
        grad_cell_weights = np.zeros_like(self.lnn.cell.weights)
        grad_cell_bias = np.zeros_like(self.lnn.cell.bias)
        grad_output_weights = np.zeros_like(self.lnn.output_weights)
        grad_output_bias = np.zeros_like(self.lnn.output_bias)
        
        # Initialize the gradient of the loss with respect to the next hidden state as zero
        grad_next_h = np.zeros(self.lnn.hidden_size)

        # Process the sequence in reverse order
        for t in reversed(range(len(input_sequence))):
            output = outputs[t]
            target = target_sequence[t]
            hidden_state = hidden_states[t]
            
            # Previous hidden state (or zeros if it's the first step)
            prev_hidden_state = hidden_states[t-1] if t > 0 else np.zeros(self.lnn.hidden_size)
            
            # 1. Backpropagate through the loss and output activation (sigmoid)
            error = output - target
            d_output = error * (output * (1 - output)) # Derivative of MSE * derivative of sigmoid
            
            # 2. Calculate gradients for the output layer
            grad_output_weights += np.outer(d_output, hidden_state)
            grad_output_bias += d_output
            
            # 3. Backpropagate gradient to the hidden state
            grad_h = np.dot(self.lnn.output_weights.T, d_output) + grad_next_h
            
            # 4. Backpropagate through the cell's ODE update and activation (tanh)
            # This is a simplified BPTT for this specific cell structure.
            # grad_h is the gradient of the loss w.r.t. state(t)
            
            # Derivative of tanh is 1 - tanh^2
            # The activated part of the cell's forward pass was tanh(W*x + b)
            # Let's assume the cell's `forward` stored the pre-activation value if needed.
            # For simplicity here, we'll approximate based on the state.
            # A full BPTT would require storing more intermediate values.
            
            # Gradient of the linear part inside the cell
            d_linear = grad_h * (1 - np.tanh(hidden_state)**2)
            
            # 5. Calculate gradients for the cell's weights and biases
            combined_input = np.concatenate((input_sequence[t], prev_hidden_state))
            grad_cell_weights += np.outer(d_linear, combined_input)
            grad_cell_bias += d_linear
            
            # 6. Update the gradient for the next (previous in time) hidden state
            grad_next_h = np.dot(self.lnn.cell.weights[:, self.lnn.input_size:].T, d_linear)

        # --- Update Parameters using Gradient Descent with Momentum ---
        
        # Calculate current learning rate with decay
        current_learning_rate = self.initial_learning_rate * np.exp(-self.decay_rate * self.iteration)
        
        # Update cell weights
        self.v_cell_weights = self.momentum * self.v_cell_weights - current_learning_rate * grad_cell_weights
        self.lnn.cell.weights += self.v_cell_weights
        
        # Update cell bias
        self.v_cell_bias = self.momentum * self.v_cell_bias - current_learning_rate * grad_cell_bias
        self.lnn.cell.bias += self.v_cell_bias
        
        # Update output weights
        self.v_output_weights = self.momentum * self.v_output_weights - current_learning_rate * grad_output_weights
        self.lnn.output_weights += self.v_output_weights
        
        # Update output bias
        self.v_output_bias = self.momentum * self.v_output_bias - current_learning_rate * grad_output_bias
        self.lnn.output_bias += self.v_output_bias

        self.iteration += 1

    def get_iteration(self) -> int:
        return self.iteration
