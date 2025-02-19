from typing import List, Tuple, Union, Optional
import numpy as np
from scipy import misc
from imp import reload

from labfuns import *
import random


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


"""
============
ASSIGNMENT 1
============
"""
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
        mask = labels == k
        W_k = W[mask]
        X_k = X[mask]
        
        W_k_sum = np.sum(W_k)
        mu[idx] = np.sum(W_k * X_k, axis=0) / W_k_sum
        
        diff = X_k - mu[idx]
        sigma[idx] = np.diag(np.sum(W_k * (diff ** 2), axis=0) / W_k_sum)

    return mu, sigma


"""
============
ASSIGNMENT 2
============
"""
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(
    labels: np.ndarray, W=None
) -> np.ndarray:
    """ Estimates and returns the class prior in X (ignore the W argument). """
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts, 1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses, 1))
    for idx, k in enumerate(classes):
        mask = labels == k
        prior[idx] = np.sum(W[mask]) / np.sum(W)
    
    return prior


# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(
    X: np.ndarray, prior: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
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

    

"""
============
ASSIGNMENT 4
============
"""
# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(
    base_classifier: Union[BayesClassifier, BoostClassifier], 
    X: np.ndarray, 
    labels: np.ndarray, 
    T: int =10
):
    # these will come in handy later on
    Npts, Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts, 1)) / float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # TODO: Fill in the rest, construct the alphas etc.
        # ==========================
        # alpha = 
        # alphas.append(alpha) # you will need to append the new alpha
        # ==========================
        
    return classifiers, alphas


# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        
        # ==========================

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)
    

