"""
Linear, Logistic, Regression Models using PyTorch
Full implementation with GPU support and proper training loops
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
from .base_model import BaseNeuralModel
from .base_model import BaseSupervisedModel


# ============================================================================
# LINEAR REGRESSION
# ============================================================================

class LinearRegressionNet(nn.Module):
    """Neural network for linear regression."""
    
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)

"""
Ordinary Least Squares (OLS) Linear Regression
Compatible with BaseSupervisedModel
"""

class LinearRegression(BaseSupervisedModel):
    """
    Linear Regression using Ordinary Least Squares (OLS).
    
    Formula:
        w = (XᵀX)^(-1) Xᵀ y
        y_pred = Xw + b

    Args:
        fit_intercept: Whether to include intercept term (default: True)
        device: Computation device
    """
    
    def __init__(self, fit_intercept: bool = True, device: Optional[torch.device] = None):
        super().__init__(device)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._XTX_inv = None  # cached for analysis (optional)
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor], 
            verbose: bool = False) -> 'LinearRegression':
        """
        Fit OLS linear regression model analytically.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) or (n_samples, 1)
            verbose: Print fitting info

        Returns:
            self: Fitted model
        """
        # Convert inputs
        X = self._to_tensor(X, dtype=torch.float32)
        y = self._to_tensor(y, dtype=torch.float32)
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze()
            if y.dim() != 1:
                raise ValueError("y must be 1D or a column vector")
        
        # Validate input
        self._validate_input(X, y)
        
        # Store original dimensions BEFORE modifying X
        n_samples, n_features = X.shape
        X_augmented = X  # Keep reference to augmented version

        # Add intercept term if needed
        if self.fit_intercept:
            ones = torch.ones((X.shape[0], 1), device=self.device)
            X_augmented = torch.cat([ones, X], dim=1)

        # Compute closed-form solution: (XᵀX)^(-1) Xᵀ y
        XT = X_augmented.T
        XTX = XT @ X_augmented
        try:
            XTX_inv = torch.linalg.inv(XTX)
        except RuntimeError:
            # Use pseudo-inverse for singular matrices
            XTX_inv = torch.linalg.pinv(XTX)
        
        w = XTX_inv @ XT @ y

        # Store coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = w[0].item()
            self.coef_ = self._to_numpy(w[1:])
        else:
            self.intercept_ = 0.0
            self.coef_ = self._to_numpy(w)

        # Cache inverse for statistical analysis
        self._XTX_inv = self._to_numpy(XTX_inv)
        
        # Update fit metadata with ORIGINAL feature dimensions
        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples
        self.is_fitted = True
        
        if verbose:
            print(f"LinearRegression fitted with {n_features} features on {n_samples} samples")
            print(f"Intercept: {self.intercept_:.4f}")
            print(f"Coefficients shape: {self.coef_.shape}")
        
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Input features (n_samples, n_features)
        Returns:
            Predicted values as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        X = self._to_tensor(X, dtype=torch.float32)
        
        # Validate feature count
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with "
                f"{self.n_features_in_} features"
            )
        
        # Convert coefficients to tensor
        coef_tensor = self._to_tensor(self.coef_, dtype=torch.float32)
        
        # Compute predictions: y = Xw + b
        y_pred = X @ coef_tensor + self.intercept_
        
        return self._to_numpy(y_pred).flatten()

    def score(self, X: Union[np.ndarray, torch.Tensor],
              y: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute R² (coefficient of determination) score.

        Args:
            X: Test features
            y: True targets
        Returns:
            R² score (1.0 is perfect prediction, 0.0 is baseline)
        """
        y_pred = self.predict(X)

        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        y = y.flatten()

        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        if ss_tot == 0:
            # Perfect prediction or constant target
            return 1.0 if ss_res == 0 else 0.0
        
        return 1.0 - (ss_res / ss_tot)

    def get_params(self):
        """Return model parameters."""
        params = super().get_params()
        params.update({
            'fit_intercept': self.fit_intercept,
            'coef_': self.coef_,
            'intercept_': self.intercept_
        })
        return params
    
    def get_residuals(self, X: Union[np.ndarray, torch.Tensor],
                      y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute residuals (errors) for given data.
        
        Args:
            X: Features
            y: True targets
        
        Returns:
            Residuals: y - y_pred
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing residuals.")
        
        y_pred = self.predict(X)
        
        if isinstance(y, torch.Tensor):
            y = self._to_numpy(y)
        
        return y.flatten() - y_pred
    
