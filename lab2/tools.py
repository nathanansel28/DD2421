from typing import Literal, Union, List, Callable
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import numpy as np
import random, math


class SuportVectorMachine:
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        kernel_type: Literal['linear', 'polynomial', 'rbf']
    ):
        self.X = X
        self.y = y
        self.kernel = self._initialize_kernel()
        self.P_matrix = self._initialize_p_matrix()


    def _initialize_kernel(self, kernel_type: str) -> Callable:
        """Initialize the kernel function based on the kernel type."""
        if kernel_type == 'linear':
            return self._func_linear
        elif kernel_type == 'polynomial':
            return self._func_polynomial
        elif kernel_type == 'rbf':
            return self._func_rbf
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")


    def _initialize_p_matrix(self) -> np.ndarray:
        """Compute the P matrix where P[i, j] = y[i] * y[j] * Kernel(X[i], X[j])."""
        n_samples = self.X.shape[0]
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            P[i, :] = self.y[i] * self.y * np.array(
                [self.kernel(self.X[i], self.X[j]) for j in range(n_samples)]
            )
        return P


    def objective_function(
        self, 
        alpha: np.ndarray
    ) -> float: 
        # TODO: assert alpha.shape == (something)
        return 0.5 * np.dot(alpha, np.dot(self.P_matrix, alpha)) - np.sum(alpha)


    def _func_linear(self, X: np.ndarray, y: np.ndarray) -> float:
        """Linear kernel: K(X, y) = X · y"""
        return np.dot(X, y)

    def _func_polynomial(self, X: np.ndarray, y: np.ndarray) -> float:
        """Polynomial kernel: K(X, y) = (X · y + 1)^degree"""
        return (np.dot(X, y) + 1) ** self.degree

    def _func_rbf(self, X: np.ndarray, y: np.ndarray) -> float:
        """Radial Basis Function (RBF) kernel: K(X, y) = exp(-gamma * ||X - y||^2)"""
        return np.exp(-self.gamma * np.linalg.norm(X - y) ** 2)



