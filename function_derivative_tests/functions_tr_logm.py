"""
Sample code automatically generated on 2023-10-22 00:15:57

by www.matrixcalculus.org

from input

d/dA tr(B'*log(A)) = B'*inv(A)

where

A is a symmetric psd matrix
B is a symmetric psd matrix

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

import scipy

def fAndG(A, B):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert B_rows == A_cols
    assert B_cols == A_rows

    functionValue = np.trace(B.dot(scipy.linalg.logm(A)))
    invA = np.linalg.inv(A)
    sqrt_invA = scipy.linalg.sqrtm(invA)
    gradient = B.dot(invA)

    return functionValue, gradient

def checkGradient(A, B):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1e-6
    d = A.shape[0]
    delta = np.random.randn(d, d)
    delta += delta.T
    f1, _ = fAndG(A + t * delta, B)
    f2, _ = fAndG(A - t * delta, B)
    f, g = fAndG(A, B)
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
    A, B = generateRandomData(dim=5)
    functionValue, gradient = fAndG(A, B)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(A, B)
