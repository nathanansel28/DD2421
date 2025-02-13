from typing import Literal, Union, List, Callable, Optional
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import numpy as np
import random, math


class SupportVectorMachine:
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        kernel_type: Literal['linear', 'polynomial', 'rbf'],
        degree: Optional[int] = None,
        sigma: Optional[float] = None,
    ):
        """
        SVM implementation from scratch. 

        Parameters
        ----------
        X : np.ndarray
            The patterns or inputs of the dataset.
        y : np.ndarray
            The target values corresponding to the patterns. 
        kernel_type : Literal['linear', 'polynomial', 'rbf']
            Choose the type of kernel used.
        degree : Optional[int] = None
            Only necessary for polynomial kernels.
        sigma : Optional[float] = None
            Only necessary for rbf kernels; controls the width of the RBF kernels.
        """
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.N: int = self.X.shape[0]
        self.alpha: np.ndarray = self._init_alpha()
        self.degree = degree if degree is not None else 2
        self.sigma = sigma if sigma is not None else 0.1
        self.kernel: Callable = self._init_kernel(kernel_type)
        self.P_matrix: np.ndarray = self._init_p_matrix()
        

    def fit(self) -> np.ndarray:
        """
        Fits the SVM model by calling `minimize` from scipy.optimize.
        Calculates the support vectors by extracting the nonzero alphas. 
        Calculates the hyperplane's bias using eq. 7.

        Returns
        -------
        self.ret['x]
            The vector which minimizes the objective function subject to the constraints.
        """
        self.ret = minimize(
            self._objective,
            self.alpha, 
            bounds=[(0, None) for b in range(self.N)],
            constraints={'type': 'eq', 'fun': self._zerofun} 
        )

        self.alpha: np.ndarray = self.ret['x']
        self.nonzero_alphas = np.where(self.alpha > 1e-5)[0]
        sv_index = self.nonzero_alphas[0]  # Take any support vector (first one here)
        support_vector = self.X[sv_index]
        support_target_value = self.y[sv_index]

        self.b: float = np.sum(
            self.alpha * self.y * np.array(
                [self.kernel(support_vector, self.X[i]) for i in range(self.N)]
            )
        ) - support_target_value

        return self.ret['x']



    def _indicator(
        self, s: np.ndarray
    ) -> bool:
        """Indicate whether or not a sample vector `s` belongs to the positive class."""
        indicator_value: float = np.sum(
            self.alpha * self.y * np.array(
                [self.kernel(s, self.X[i]) for i in range(self.N)]
            ) - self.b
        )
        return bool(indicator_value)


    def predict(
        self, X_predict: np.ndarray
    ) -> np.ndarray:
        """Creates predictions on X_predict using the indicator function."""
        return np.array([self._indicator(sub_arr) for sub_arr in X_predict])


    def evaluate(
        self, y_actual: np.ndarray, y_predict: np.ndarray
    ) -> float:
        """Returns the accuracy of the predictions."""
        assert y_actual.shape == y_predict.shape
        return float(np.sum(y_actual == y_predict) / y_actual.size)


    def _objective(
        self, alpha: np.ndarray
    ) -> float: 
        """Function which implements the objective function in eq. 4."""
        return 0.5 * np.dot(alpha, np.dot(self.P_matrix, alpha)) - np.sum(alpha)


    def _zerofun(
        self, alpha: np.ndarray
    ) -> float:
        """Function which implements the equality constraint in eq. 10 (sum alpha * t = 0)."""
        return np.sum(np.dot(alpha, self.y))


    def _init_alpha(self) -> np.ndarray: 
        """Initialize the alpha with zeroes.""" 
        return np.zeros(self.N)


    def _init_kernel(self, kernel_type: str) -> Callable:
        """Initialize the kernel function based on the kernel type."""
        if kernel_type == 'linear':
            return self._func_linear
        elif kernel_type == 'polynomial':
            return self._func_polynomial
        elif kernel_type == 'rbf':
            return self._func_rbf
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")


    def _init_p_matrix(self) -> np.ndarray:
        """Compute the P matrix where P[i, j] = y[i] * y[j] * Kernel(X[i], X[j])."""
        n_samples = self.X.shape[0]
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            P[i, :] = self.y[i] * self.y * np.array(
                [self.kernel(self.X[i], self.X[j]) for j in range(n_samples)]
            )
        return P
    

    def _func_linear(self, X: np.ndarray, y: np.ndarray) -> float:
        """Linear kernel: K(X, y) = X · y"""
        return np.dot(X, y)

    def _func_polynomial(self, X: np.ndarray, y: np.ndarray) -> float:
        """Polynomial kernel: K(X, y) = (X · y + 1)^degree"""
        return (np.dot(X, y) + 1) ** self.degree

    def _func_rbf(self, X: np.ndarray, y: np.ndarray) -> float:
        """Radial Basis Function (RBF) kernel: K(X, y) = exp(-||X - y||^2 / 2 sigma^2)"""
        return np.exp(
            (-np.linalg.norm(X - y) ** 2) / (2 * self.sigma**2)
        )



