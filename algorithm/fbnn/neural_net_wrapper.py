import numpy as np
from typing import List, Optional
import warnings
import logging

from config.sdt_constants import DEFAULTITER

# NOTE: This implementation requires the scikit-learn library.
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    logging.error("ERROR: scikit-learn is not installed. Please run 'pip install scikit-learn'")
    class MLPRegressor:
        def __init__(self, *args, **kwargs): pass
        def predict(self, X): return np.zeros(X.shape[0])
    class ConvergenceWarning(UserWarning): pass

class NeuralNet:
    """
    High-precision wrapper for scikit-learn's Multi-layer Perceptron (MLP) regressor.
    
    This class specializes the MLPRegressor for telemetry modeling by using the 
    L-BFGS quasi-Newton optimizer, which is well-suited for smaller, 
    low-dimensional telemetry datasets. It provides robust weight management, 
    automatic recovery from solver failures, and a flat-array interface for 
    weight serialization.
    """
    
    def __init__(self, activation_func: str, structure: List[int], max_iter: int = DEFAULTITER, tol: float = 1.0e-6, warm_start: bool = True, random_state: int = 0, alpha: float = 0.01):
        """
        Initializes the Neural Network model.

        Args:
            activation_func (str): The activation function ('TANH' or 'RELU').
            structure (List[int]): The network topology, e.g., [inputs, hidden1, outputs].
            max_iter (int): Maximum number of iterations for the L-BFGS solver.
            tol (float): Tolerance for the optimization stop criteria.
            warm_start (bool): If True, reuse the solution of the previous call 
                to fit as initialization.
            random_state (int): Seed for the random number generator.
            alpha (float): L2 regularization penalty strength.
        """
        hidden_layers = tuple(structure[1:-1])
        activation = 'tanh' if activation_func == "TANH" else 'relu'
        
        self.input_size = structure[0]
        self.output_size = structure[-1]
        self.structure = structure
        self._weight_dim = 0
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver='lbfgs',
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            random_state=random_state
        )
        self._is_fitted = False

    def get_weight_dim(self) -> int:
        """
        Calculates the total number of weights and biases in the network.

        Returns:
            int: The total parameter count.
        """
        if not hasattr(self, '_weight_dim') or self._weight_dim == 0:
            layer_sizes = [self.input_size] + list(self.model.hidden_layer_sizes) + [self.output_size]
            dim = 0
            for i in range(len(layer_sizes) - 1):
                dim += layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
            self._weight_dim = dim
        return self._weight_dim

    def set_weight(self, weights: np.ndarray):
        """
        Reconstructs the internal model state from a flat array of weights.
        
        This method performs a 'dummy' fit if the model was never initialized 
        to ensure internal scikit-learn structures are built before injecting 
        the provided weights.

        Args:
            weights (np.ndarray): A 1D array containing the serialized weights and biases.
        """
        if not self._is_fitted:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                # Use small random values for dummy fit to avoid singular Hessian matrices
                rng = np.random.default_rng(1)
                dummy_X = rng.standard_normal((5, self.input_size)) * 0.1
                dummy_y = rng.standard_normal(5) * 0.1
                self.model.fit(dummy_X, dummy_y)

        # Map flat array to scikit-learn coefs_ (weights) and intercepts_ (biases)
        layer_units = [self.input_size] + list(self.model.hidden_layer_sizes) + [self.output_size]
        self.model.coefs_ = []
        self.model.intercepts_ = []
        start = 0
        
        # Inject weights
        for i in range(len(layer_units) - 1):
            end = start + layer_units[i] * layer_units[i+1]
            self.model.coefs_.append(weights[start:end].reshape((layer_units[i], layer_units[i+1])))
            start = end
            
        # Inject biases
        for i in range(len(layer_units) - 1):
            end = start + layer_units[i+1]
            self.model.intercepts_.append(weights[start:end])
            start = end
            
        self._is_fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Executes model training using the L-BFGS optimizer.
        
        It includes an automatic recovery mechanism: if the solver fails (e.g., 
        singular matrix), it will automatically retry with a 10x higher 
        regularization (alpha) to stabilize the optimization.

        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target values array.
        """
        if X.ndim == 1: 
            X = X.reshape(-1, 1)
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                self.model.fit(X, y)
        except Exception as e:
            logging.warning(f"NeuralNet: L-BFGS solver failed ({e}). Attempting recovery with alpha={self.model.alpha * 10}.")
            original_alpha = self.model.alpha
            self.model.alpha = original_alpha * 10
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    self.model.fit(X, y)
            except Exception:
                self.model.alpha = original_alpha 
                raise
            
        self._is_fitted = True

    def get_output(self, network_input: np.ndarray) -> np.ndarray:
        """
        Performs inference using the trained network.

        Args:
            network_input (np.ndarray): Input features (single vector or batch).

        Returns:
            np.ndarray: Predicted values.
        """
        if not self._is_fitted:
            return np.array([0.0])

        if network_input.ndim == 1:
            network_input = network_input.reshape(-1, 1)
            
        return self.model.predict(network_input)

    def get_network_struct(self) -> List[int]:
        """Returns the current network topology as a list of layer sizes."""
        return [self.input_size] + list(self.model.hidden_layer_sizes) + [self.output_size]
