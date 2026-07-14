import logging
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler


def run_ridge_regression(x_train: np.ndarray, y_train: np.ndarray, alpha: float, fit_intercept: bool = True, scale_data: bool = True) -> np.ndarray:
    """
    Executes a Ridge Regression (L2-regularized linear regression) on the dataset.
    
    This function uses a scikit-learn pipeline to optionally scale the data 
    and then solves the regularized least-squares problem: (X'X + αI)w = X'y. 
    Crucially, it transforms the resulting weights back into physical units so 
    they can be applied directly to un-scaled input features during prediction.

    Args:
        x_train (np.ndarray): 2D array of input features [samples, features].
        y_train (np.ndarray): 1D array of target telemetry values.
        alpha (float): Regularization strength (must be >= 0).
        fit_intercept (bool): If True, calculates the bias/intercept term.
        scale_data (bool): If True, normalizes inputs to zero mean and unit 
            variance before fitting. Recommended for numerical stability.

    Returns:
        np.ndarray: A 1D array of optimized coefficients: [intercept, w1, w2, ..., wn].
        
    Raises:
        ValueError: If input dimensions are inconsistent.
    """
    RIDGE = "RIDGE"

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("The number of samples in X and y do not match.")
    
    # Ensure alpha is not too small to avoid numerical instability
    # If alpha is 0, Ridge becomes OLS which is prone to singular matrix errors.
    safe_alpha = max(alpha, 1e-8)

    # Clean up near-zero features which can cause division issues in matmul
    x_processed = np.where(np.abs(x_train) < 1e-15, 0.0, x_train)

    # --- 1. Model Training ---
    # Using 'svd' solver as it's the most robust for ill-conditioned matrices
    ridge_model = Ridge(alpha=safe_alpha, fit_intercept=fit_intercept, solver='svd')

    if scale_data:
        model = make_pipeline(StandardScaler(), ridge_model)
    else:
        model = Pipeline([('ridge', ridge_model)])
    try:
        with np.errstate(all="ignore"):
            model.fit(x_processed, y_train)
    except Exception as e:
        logging.error(f"[{RIDGE}] Error during model fit: {e}")
        # Robust fallback: intercept = mean(y), weights = 0
        num_features = x_train.shape[1]
        return np.concatenate(([np.mean(y_train)], np.zeros(num_features)))

    # --- 2. Extract and Un-scale Coefficients ---
    # coefficients are calculated as: w_physical = w_normalized / std_dev
    trained_ridge = model.named_steps['ridge']
    
    raw_coefs = trained_ridge.coef_
    raw_intercept = trained_ridge.intercept_

    if scale_data:
        scaler = model.named_steps['standardscaler']
        means = scaler.mean_
        scales = scaler.scale_

        # Avoid division by zero if a feature has zero variance (constant)
        safe_scales = np.where(scales < 1e-12, 1.0, scales)
        real_coefs = np.where(scales < 1e-12, 0.0, raw_coefs / safe_scales)
        
        # Adjust intercept to account for the mean subtraction during scaling
        real_intercept = raw_intercept - np.sum(real_coefs * means)
    else:
        real_coefs = raw_coefs
        real_intercept = raw_intercept

    # --- 3. Evaluating the Model ---
    try:
        y_pred = model.predict(x_processed)
        rss = np.sum((y_train - y_pred) ** 2)
        r_squared = r2_score(y_train, y_pred)
        logging.info(f"[{RIDGE}] Fit complete. RSS: {rss:.4f}, R-squared: {r_squared:.4f}")
    except Exception as e:
        logging.error(f"[{RIDGE}] Evaluation failed: {e}")

    # Return in standardized format: [intercept, w1, w2, ...]
    return np.concatenate(([real_intercept], real_coefs))


if __name__ == '__main__':
    # --- Standalone Testing ---
    np.random.seed(0)
    n_samples, n_features = 50, 5
    X = np.random.rand(n_samples, n_features)
    y = 5.0 + 2.0 * X[:, 0] + np.random.randn(n_samples) * 0.1

    print("--- Running Ridge Regression test ---")
    params = run_ridge_regression(X, y, alpha=0.1, scale_data=True)
    print(f"Resulting Parameters: {params}")
