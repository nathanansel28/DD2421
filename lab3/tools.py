# %%
from typing import List, Tuple, Union, Optional
import numpy as np
from scipy import misc
from imp import reload

from labfuns import *
import random


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    """ Estimates and returns the class prior in X (ignore the W argument). """
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))
    prior = np.array([
        len(labels[labels==k]) / Npts for k in np.unique(labels)
    ])
    
    return prior


# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(
    X: np.ndarray, labels: np.ndarray, W=None
) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculates and returns the ML-estimates of the mu and sigma of a dataset. """

    assert(X.shape[0]==labels.shape[0])
    classes, inverse = np.unique(labels, return_inverse=True)
    C = len(classes)
    N, d = np.shape(X)

    if W is None:
        W = np.ones((N, 1)) / float(N)

    mu = np.zeros((C, d))
    sigma = np.zeros((C, d, d))

    counts = np.zeros(C)
    np.add.at(mu, inverse, X)
    np.add.at(counts, inverse, 1)
    mu /= counts[:, None]

    for idx, k in enumerate(classes):
        X_k = X[labels == k]
        N_k = X_k.shape[0]
        diff = X_k - mu[idx]
        sigma[idx] = (diff.T @ diff) / N_k

    return mu, sigma



# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):
    """
    Computes the discriminant function values for all classes and data points, 
    and classifies each point to belong to the max discriminant value. 
    """

    arrays = [
        (
            - 1/2 * np.log(np.linalg.det(sigma[k])) 
            - 1/2 * np.einsum('ij,jk,ik->i', (X - mu[k]), np.linalg.inv(sigma[k]), (X - mu[k]))
            + np.log(prior[k])
        ) for k in range(prior.shape[0])
    ]
    
    log_prob = np.vstack(arrays)
    h = np.argmax(log_prob, axis=0) 
    return h
