"""
Analyze tensor from make_tensor.
"""
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition.candecomp_parafac import normalize_factors
from tensorly.metrics.regression import variance as tl_var


def z_score_values(A, cell_dim):
    ''' Function that takes in the values tensor and z-scores it. '''
    assert cell_dim < tl.ndim(A)
    convAxes = tuple([i for i in range(tl.ndim(A)) if i != cell_dim])
    convIDX = [None] * tl.ndim(A)
    convIDX[cell_dim] = slice(None)

    sigma = tl.tensor(np.std(tl.to_numpy(A), axis=convAxes))
    return A / sigma[tuple(convIDX)]


def R2X(reconstructed, original):
    ''' Calculates R2X of two tensors. '''
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)


def perform_decomposition(tensor, r, weightFactor=2):
    ''' Apply z-scoring and perform PARAFAC decomposition. '''
    factors = non_negative_parafac(tensor, r, tol=1.0E-9, n_iter_max=10000)
    factors, weights = normalize_factors(factors)  # Position 0 is factors. 1 is weights.
    factors[weightFactor] *= weights[np.newaxis, :]  # Put weighting in designated factor
    return factors


def find_R2X(values, factors):
    '''Compute R2X from parafac. Note that the inputs values and factors are in numpy.'''
    return R2X(tl.kruskal_to_tensor(factors), values)
