import numpy as np
import logging
from typing import List, Optional, Protocol

# Define a Protocol for the basis function interface
class Function(Protocol):
    def func(self, x: float) -> np.ndarray:
        """Returns the values of the basis functions at x."""
        ...

class LSF:
    """
    Provides a general-purpose Least Squares Fit (LSF) algorithm.
    Optimized using NumPy for high performance and numerical stability.
    """
    CONTEXT = "LSF"

    def __init__(self, weighted: bool, param_num: int):
        """
        Constructs a new LSF solver.
        :param weighted: True if weighted least squares should be used.
        :param param_num: The number of basis functions (parameters).
        """
        self.weighted = weighted
        self.param_num = param_num

    def lfit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray, n: int, my: int,
             func: Function) -> Optional[dict]:
        """
        Performs a least-squares fit for one or more datasets.
        
        :param x: Array of independent variable values (size n).
        :param y: 2D array where each row is a set of dependent variable values (shape my, n).
        :param weight: Array of weights for each data point (size n).
        :param n: Number of data points.
        :param my: Number of datasets to fit.
        :param func: Object providing the basis functions.
        :return: A dictionary containing 'coefficients', 'covariance', and 'chisq', or None on failure.
        """
        try:
            # 1. Build the Design Matrix A
            # A has shape (n, m) where m is the number of parameters
            # Instead of looping, we try to vectorize the basis function evaluation
            # If func.func only handles scalars, we use a list comprehension.
            
            # Extract basis function values for each x
            # Note: We assume the basis functions returned by func() are aligned with param_num
            basis_values = np.array([func.func(val) for val in x])
            # Handle index 0 if the Java interface used 1-based indexing
            if basis_values.shape[1] > self.param_num:
                A = basis_values[:, 1:self.param_num + 1]
            else:
                A = basis_values

            results_a = np.zeros((self.param_num, my))
            results_chisq = np.zeros(my)
            covariance = None

            # 2. Apply Weights
            if self.weighted:
                # WLS is solved by transforming to OLS: (W^1/2 A) x = (W^1/2 y)
                sqrt_w = np.sqrt(weight)[:, np.newaxis]
                A_weighted = A * sqrt_w
            else:
                A_weighted = A

            # 3. Solve Least Squares for each dataset
            for k in range(my):
                target_y = y[k, :]
                
                if self.weighted:
                    target_y_weighted = target_y * np.sqrt(weight)
                else:
                    target_y_weighted = target_y

                # Solve: A_weighted * a = target_y_weighted
                # rcond=None lets numpy determine the cutoff for small singular values
                a, residuals, rank, s = np.linalg.lstsq(A_weighted, target_y_weighted, rcond=None)
                
                results_a[:, k] = a
                
                # Chi-squared is the sum of squared residuals
                if residuals.size > 0:
                    results_chisq[k] = residuals[0]
                else:
                    # Manually calculate if residuals not returned (rank deficient)
                    fit_y = A_weighted @ a
                    results_chisq[k] = np.sum((target_y_weighted - fit_y)**2)

            # 4. Calculate Covariance Matrix
            # Covariance matrix is (A^T W A)^-1
            # We use the pseudo-inverse for stability
            try:
                ATA = A_weighted.T @ A_weighted
                covariance = np.linalg.pinv(ATA)
            except np.linalg.LinAlgError:
                logging.error(f"{self.CONTEXT}: Singular matrix encountered during covariance calculation.")

            return {
                'coefficients': results_a,
                'covariance': covariance,
                'chisq': results_chisq
            }

        except Exception as e:
            logging.error(f"{self.CONTEXT}: Least Squares Fitting failed: {e}")
            return None

    @staticmethod
    def solve_linear_system(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Utility method to solve A*X = B."""
        return np.linalg.solve(A, B)
