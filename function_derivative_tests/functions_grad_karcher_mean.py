"""
from input

d/dA \|log(A)\|^2

where

A is a symmetric psd matrix
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import scipy

def symm(A):
    return 0.5*(A + A.T)

def fAndG(A):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2

    logA = scipy.linalg.logm(A)
    functionValue = np.trace(logA.dot(logA))
    # functionValue = np.linalg.norm(logA, 2) ** 2
    invA = np.linalg.inv(A)
    # sqrt_invA = scipy.linalg.sqrtm(invA)
    gradient = 2 * logA.dot(invA)

    return functionValue, gradient

def checkGradient(A):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1e-10
    d = A.shape[0]
    delta = np.random.randn(d, d)
    delta = symm(delta)
    # delta = np.diag(np.diag(delta))
    f1, _ = fAndG(A + t * delta)
    f2, _ = fAndG(A - t * delta)
    f, g = fAndG(A)
    print('approximation error',
            abs( (f1 - f2) / (2*t) - np.trace( g.T.dot(delta) ) ))

def generateRandomData(dim):
    A = np.random.randn(dim, dim)
    A = A.T.dot(A)
    A = symm(A) + np.eye(dim)  # make it symmetric and psd

    # return np.diag(np.diag(A))
    return A

if __name__ == '__main__':
    A = generateRandomData(dim=3)
    functionValue, gradient = fAndG(A)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A)
