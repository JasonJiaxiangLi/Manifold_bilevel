"""
Sample code automatically generated on 2023-10-22 00:39:50

by www.matrixcalculus.org

from input

d/dX tr(A'*inv(X)) = 0.5*(-(inv(X)'*A*inv(X)'+inv(X)*A'*inv(X)))

where

A is a symmetric matrix
X is a symmetric matrix

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(A, X):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(X, np.ndarray)
    dim = X.shape
    assert len(dim) == 2
    X_rows = dim[0]
    X_cols = dim[1]
    assert X_rows == A_rows
    assert X_cols == A_cols

    T_0 = np.linalg.inv(X)
    functionValue = np.trace((A).dot(T_0))
    gradient = (0.5 * -(((T_0.T).dot(A)).dot(T_0.T) + ((T_0).dot(A.T)).dot(T_0)))

    return functionValue, gradient

def checkGradient(A, X):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    d = A.shape[0]
    delta = np.random.randn(d, d)
    delta += delta.T
    f1, _ = fAndG(A, X + t * delta)
    f2, _ = fAndG(A, X - t * delta)
    f, g = fAndG(A, X)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData(dim):
    A = np.random.randn(dim, dim)
    A = A.T.dot(A)
    A = 0.5 * (A + A.T) + np.eye(dim)  # make it symmetric and psd
    B = np.random.randn(dim, dim)
    B = B.T.dot(B)
    B = 0.5 * (B + B.T) + np.eye(dim) # make it symmetric and psd

    return A, B

if __name__ == '__main__':
    A, X = generateRandomData(dim=5)
    functionValue, gradient = fAndG(A, X)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A, X)
