"""
Sample code automatically generated on 2023-10-22 00:15:57

by www.matrixcalculus.org

from input

d/dA tr(C * log(P X Q))

where

X is a symmetric psd matrix
P, Q, C are cosntant symmetric psd matrix
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy

def symm(A):
    return (A + A.T) / 2

def fAndG(X, P, Q, C):
    functionValue = np.trace(C.dot(P.dot(X).dot(Q)))
    d = X.shape[0]
    gradient = np.zeros_like(X)
    zeros = np.zeros_like(X)
    temp1 = np.block([[zeros, C], [zeros, zeros]])
    temp2 = np.block([[P, zeros], [zeros, P]])
    temp4 = np.block([[Q, zeros], [zeros, Q]])
    for i in range(d):
        for j in range(d):
            Eij = np.zeros_like(X)
            Eij[i, j] = 1
            temp3 = np.block([[X, Eij], [zeros, X]])
            gradient[i, j] = np.trace(temp1.T.dot(scipy.linalg.logm(temp2.dot(temp3).dot(temp4))))

    return functionValue, gradient

def checkGradient(X, P, Q, C):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1e-6
    d = X.shape[0]
    delta = np.random.randn(d, d)
    delta = symm(delta)
    f1, _ = fAndG(X + t * delta, P, Q, C)
    f2, _ = fAndG(X - t * delta, P, Q, C)
    _, g = fAndG(X, P, Q, C)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData(dim):
    def genOneData(dim):
        A = np.random.randn(dim, dim)
        A = A.T.dot(A)
        A = 0.5 * (A + A.T) + np.eye(dim)  # make it symmetric and psd
        return A
    
    X, P, Q, C = genOneData(dim), genOneData(dim), genOneData(dim), genOneData(dim)

    return X, P, Q, C

if __name__ == '__main__':
    X, P, Q, C = generateRandomData(dim=5)
    functionValue, gradient = fAndG(X, P, Q, C)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(X, P, Q, C)
