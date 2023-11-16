"""
from input

d/dX \|log(A^{-1/2} X A^{-1/2})\|^2

where

X is a symmetric psd matrix
A is a symmetric psd matrix
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy

def symm(X):
    return 0.5*(X + X.T)

def fAndG(X, A):
    assert isinstance(X, np.ndarray)
    dim = X.shape
    assert len(dim) == 2
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2

    inv_sqrt_A = np.linalg.inv(scipy.linalg.sqrtm(A))
    temp = scipy.linalg.logm(inv_sqrt_A.dot(X).dot(inv_sqrt_A))
    functionValue = np.linalg.norm(temp) ** 2
    # sqrt_invA = scipy.linalg.sqrtm(invA)
    inv_sqrt_X = np.linalg.inv(scipy.linalg.sqrtm(X))
    gradient = 2 * np.linalg.inv(X).dot(scipy.linalg.sqrtm(A)).dot(temp).dot(inv_sqrt_A)

    return functionValue, gradient

def checkGradient(X, A):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1e-10
    d = A.shape[0]
    delta = np.random.randn(d, d)
    delta = symm(delta)  # random tangent vec (symmetric)

    # delta = np.diag(np.diag(delta))
    f1, _ = fAndG(X + t * delta, A)
    f2, _ = fAndG(X - t * delta, A)
    _, g = fAndG(X, A)
    print('approximation error',
            abs( (f1 - f2) / (2*t) - np.trace( g.T.dot(delta) ) ))

def generateRandomData(dim):
    A = np.random.randn(dim, dim)
    A = A.T.dot(A)
    A = symm(A) + np.eye(dim)  # make it symmetric and psd

    X = np.random.randn(dim, dim)
    X = X.T.dot(X)
    X = symm(X) + np.eye(dim)  # make it symmetric and psd

    # return np.diag(np.diag(A))
    return X, A

if __name__ == '__main__':
    X, A = generateRandomData(dim=3)
    functionValue, gradient = fAndG(X, A)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(X, A)
