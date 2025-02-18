from typing import Literal, Union, List, Callable, Optional, Tuple
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
        randomize_alpha: bool=False
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
        self.alpha: np.ndarray = self._init_alpha(randomize_alpha)
        self.degree = degree if degree is not None else 2
        self.sigma = sigma if sigma is not None else 0.1
        self.kernel: Callable = self._init_kernel(kernel_type)
        self.P_matrix: np.ndarray = self._init_p_matrix()
        

    def fit(
        self, slack_value: Optional[float]=None
    ) -> np.ndarray:
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
            bounds=[(0, slack_value) for b in range(self.N)],
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
            )
        ) - self.b
        return 1 if indicator_value > 0 else -1 


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


    def plot_decision_boundary(
        self, 
        inputs: np.ndarray = None,
        targets: np.ndarray = None,
        grid_size: int = 200,
        fig_size: Tuple[float, float] = (12, 8),
        plot_title: str='Dataset with SVM Linear Decision Boundary',
        save_path: Optional[str] = None
    ) -> None:
        inputs = self.X if inputs is None else inputs
        targets = self.y if targets is None else targets

        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_size), 
            np.linspace(y_min, y_max, grid_size)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid_points)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=fig_size)
        plt.contourf(xx, yy, Z, alpha=0.3, levels=np.linspace(Z.min(), Z.max(), 3), cmap='coolwarm')
        plt.scatter(inputs[targets == 1][:, 0], inputs[targets == 1][:, 1], color='red', label='Class A (+1)', edgecolor='k')
        plt.scatter(inputs[targets == -1][:, 0], inputs[targets == -1][:, 1], color='blue', label='Class B (-1)', edgecolor='k')

        plt.scatter(self.X[self.nonzero_alphas][:, 0], self.X[self.nonzero_alphas][:, 1], 
                    s=100, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

        plt.xlabel('Input Feature 1')
        plt.ylabel('Input Feature 2')
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()



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


    def _init_alpha(self, randomize: bool=False) -> np.ndarray: 
        """Initialize the alpha with zeroes OR with random values between (-1, 1).""" 
        if randomize:
            np.random.seed(42)
            return np.random.uniform(-1, 1, self.N)
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



def generate_dataset(
    size: int = 20,
    classA_mean_1: List[float] = [1.5, 0.5], 
    classA_mean_2: List[float] = [-1.5, 0.5], 
    classA_std: float = 0.2, 
    classB_mean: List[float] = [0.0, -0.5],
    classB_std: float = 0.2, 
    fig_size: Tuple[int, int] = (10, 6),
    seed: int = 100
) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Generates a dataset drawn from a 2D normal distsribution according to the instructions. 

    Returns
    -------
    Tuple[inputs, targets]
        A tuple consisting the inputs and the targets. 
    """
    np.random.seed(seed)
    classA = np.concatenate((
        np.random.randn(size//2, 2) * classA_std + classA_mean_1,
        np.random.randn(size//2, 2) * classA_std + classA_mean_2
    ))

    np.random.seed(seed)
    classB = np.random.randn(size, 2) * classB_std + classB_mean

    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((
        np.ones(classA.shape[0]),
        -np.ones(classB.shape[0])
    ))

    np.random.seed(seed)
    N = inputs.shape[0]
    permute = list(range(N))
    random.shuffle(permute)

    inputs = inputs[permute, :]
    targets = targets[permute]

    plt.figure(figsize=fig_size)
    plt.scatter(inputs[targets == 1][:, 0], inputs[targets == 1][:, 1], color='blue', label='Class A (+1)')
    plt.scatter(inputs[targets == -1][:, 0], inputs[targets == -1][:, 1], color='red', label='Class B (-1)')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Input Feature 2')
    plt.title('Dataset Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

    return inputs, targets


def generate_nonlinear_dataset(
    size: int = 100,
    inner_radius: float = 0.5,
    outer_radius: float = 1.5,
    noise: float = 0.1,
    fig_size: Tuple[int, int] = (10, 6),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a non-linear dataset where Class A (positive) is inside Class B (negative),
    resembling concentric circles.

    Args:
        size (int): Total number of samples (split equally between classes).
        inner_radius (float): Radius of Class A (inner circle).
        outer_radius (float): Radius of Class B (outer ring).
        noise (float): Standard deviation of noise added to the points.
        fig_size (Tuple[int, int]): Size of the plot.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Inputs and targets.
    """
    np.random.seed(seed)
    num_per_class = size // 2

    # Generate Class A (inner circle)
    angles_A = np.random.uniform(0, 2 * np.pi, num_per_class)
    radii_A = np.random.normal(inner_radius, noise, num_per_class)
    classA = np.column_stack((radii_A * np.cos(angles_A), radii_A * np.sin(angles_A)))

    # Generate Class B (outer circle)
    angles_B = np.random.uniform(0, 2 * np.pi, num_per_class)
    radii_B = np.random.normal(outer_radius, noise, num_per_class)
    classB = np.column_stack((radii_B * np.cos(angles_B), radii_B * np.sin(angles_B)))

    # Combine datasets
    inputs = np.vstack((classA, classB))
    targets = np.hstack((np.ones(num_per_class), -np.ones(num_per_class)))

    # Shuffle dataset
    np.random.seed(seed)
    permute = np.random.permutation(size)
    inputs, targets = inputs[permute], targets[permute]

    # Plot dataset
    plt.figure(figsize=fig_size)
    plt.scatter(inputs[targets == 1][:, 0], inputs[targets == 1][:, 1], color='blue', label='Class A (+1)')
    plt.scatter(inputs[targets == -1][:, 0], inputs[targets == -1][:, 1], color='red', label='Class B (-1)')
    plt.xlabel('Input Feature 1')
    plt.ylabel('Input Feature 2')
    plt.title('Non-Linear Dataset: Concentric Circles')
    plt.legend()
    plt.grid(True)
    plt.show()

    return inputs, targets
