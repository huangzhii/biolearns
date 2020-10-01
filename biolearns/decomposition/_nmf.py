# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Wed Feb 19 13:20:25 2020
# Author: Zhi Huang, Purdue University
#
# This is a concise version rewrite from sklearn_decomposition_nmf.
#
# The original code came with the following disclaimer:
#
# This software is provided "as-is".  There are no expressed or implied
# warranties of any kind, including, but not limited to, the warranties
# of merchantability and fitness for a given application.  In no event
# shall Zhi Huang be liable for any direct, indirect, incidental,
# special, exemplary or consequential damages (including, but not limited
# to, loss of use, data or profits, or business interruption) however
# caused and on any theory of liability, whether in contract, strict
# liability or tort (including negligence or otherwise) arising in any way
# out of the use of this software, even if advised of the possibility of
# such damage.
#
from typing import Callable, Iterator, List, Optional, Tuple, Union, Any, Iterable
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.utils import check_random_state, check_array
from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from sklearn.utils.extmath import safe_sparse_dot
import copy

from math import sqrt
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_non_negative
from lifelines.utils import concordance_index
from ..survival import newton_rhapson_for_efron_model
import time
import warnings
import logging
EPSILON = np.finfo(np.float32).eps
def norm(x):
    """Dot product-based Euclidean norm implementation
    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))

    
def _initialize_nmf(X, n_components, init=None, eps=1e-6,
                    random_state=None):
    """Algorithms for NMF initialization.
    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix to be decomposed.
    n_components : integer
        The number of components desired in the approximation.
    init :  None | 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
        Method used to initialize the procedure.
        Default: None.
        Valid options:
        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)
        - 'custom': use custom matrices W and H
    eps : float
        Truncate all values less then this in output to zero.
    random_state : int, RandomState instance, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    W : array-like, shape (n_samples, n_components)
        Initial guesses for solving X ~= WH
    H : array-like, shape (n_components, n_features)
        Initial guesses for solving X ~= WH
    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (init is not None and init != 'random'
            and n_components > min(n_samples, n_features)):
        raise ValueError("init = '{}' can only be used when "
                         "n_components <= min(n_samples, n_features)"
                         .format(init))

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features).astype(X.dtype,
                                                             copy=False)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype,
                                                            copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H
def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix
    Y : array-like
        Second matrix
    """
    return np.dot(X.ravel(), Y.ravel())

def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.
    Faster than norm(x) ** 2.
    Parameters
    ----------
    x : array_like
    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order='K')
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn('Array type is integer, np.dot may overflow. '
                      'Data should be float type to avoid this issue',
                      UserWarning)
    return np.dot(x, x)

def calcuate_Frobenius_norm(X, W, H, square_root=False):
    """Compute the beta-divergence of X and dot(W, H).

    Parameters
    ----------
    X : float or array-like, shape (n_samples, n_features)

    W : float or dense array-like, shape (n_samples, n_components)

    H : float or dense array-like, shape (n_components, n_features)

    Returns
    -------
        res : float
            Frobenius norm of X and np.dot(W, H)
    """

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    # Avoid the creation of the dense np.dot(W, H) if X is sparse.
    if sp.issparse(X):
        norm_X = np.dot(X.data, X.data)
        norm_WH = trace_dot(np.dot(np.dot(W.T, W), H), H)
        cross_prod = trace_dot((X * H.T), W)
        res = (norm_X + norm_WH - 2. * cross_prod) / 2.
    else:
        res = squared_norm(X - np.dot(W, H)) / 2.

    if square_root:
        return np.sqrt(res * 2)
    else:
        return res

def _multiplicative_update_w(X, W, H, HHt=None, XHt=None, update_H=True):
    """update W in Multiplicative Update NMF"""
    # Numerator
    if XHt is None:
        XHt = safe_sparse_dot(X, H.T)
    if update_H:
        # avoid a copy of XHt, which will be re-computed (update_H=True)
        numerator = XHt
    else:
        # preserve the XHt, which is not re-computed (update_H=False)
        numerator = XHt.copy()

    # Denominator
    if HHt is None:
        HHt = np.dot(H, H.T)
    denominator = np.dot(W, HHt)
    denominator[denominator == 0] = EPSILON
    numerator /= denominator
    delta_W = numerator
    return delta_W, HHt, XHt

def _multiplicative_update_w_orth(X, W, H, HHt=None, XHt=None, sigma=0):
    '''
        Implemented based on equation (18) from:
        Mirzal, Andri. "A convergent algorithm for orthogonal nonnegative matrix factorization."
        Journal of Computational and Applied Mathematics 260 (2014): 149-166.
    '''
    if XHt is None:
        XHt = safe_sparse_dot(X, H.T)
        
    numerator = XHt + sigma*W
    # Denominator
    if HHt is None:
        HHt = np.dot(H, H.T)
    # ONMF on W
    denominator = np.dot(W, HHt) + sigma * W.dot(W.T).dot(W)
    denominator[denominator == 0] = EPSILON
    numerator /= denominator
    delta_W = numerator
    
#    # ONMF on W
#    denominator = W.dot(W.T).dot(X).dot(H.T) # Ding et al. (2006) Orthogonal Nonnegative Matrix Tri-factorizations for Clustering
#    delta_W = np.sqrt(numerator)
    
    return delta_W, HHt, XHt


def _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma):
    """update H in Multiplicative Update NMF"""
    numerator = safe_sparse_dot(W.T, X)
    denominator = np.dot(np.dot(W.T, W), H)
    denominator[denominator == 0] = EPSILON
    numerator /= denominator
    delta_H = numerator
    return delta_H

def _update_coordinate_descent(X, W, Ht, shuffle,
                               random_state):
    """Helper function for _fit_coordinate_descent

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...)

    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, permutation)



def NMF(X, n_components, solver = 'cd', max_iter=1000, tol=1e-6, update_H = True, random_state=None, shuffle=False, verbose=0):
    '''
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Constant input matrix.

    W : array-like, shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like, shape (n_components, n_features)
        Initial guess for the solution.
    '''
    W, H = _initialize_nmf(X, n_components, init = 'random', random_state=random_state)
    
    if solver == 'mu':
        # used for the convergence criterion
        error_at_init = calcuate_Frobenius_norm(X, W, H, square_root=True)
        previous_error = error_at_init
        
        start_time = time.time()
        HHt, XHt = None, None
        for n_iter in range(1, max_iter + 1):
            # update W
            # HHt and XHt are saved and reused if not update_H
            delta_W, HHt, XHt = _multiplicative_update_w(X, W, H, HHt, XHt, update_H = update_H)
            W *= delta_W
            
            # update H
            if update_H:
                delta_H = _multiplicative_update_h(X, W, H)
                H *= delta_H
                # These values will be recomputed since H changed
                HHt, XHt = None, None
        
            # test convergence criterion every 10 iterations
            if tol > 0 and n_iter % 10 == 0:
                error = calcuate_Frobenius_norm(X, W, H, square_root=True)
                if verbose:
                    iter_time = time.time()
                    print("Epoch %02d reached after %.3f seconds, error: %f" %
                          (n_iter, iter_time - start_time, error))
                if (previous_error - error) / error_at_init < tol:
                    break
                previous_error = error
    
        # do not print if we have already printed in the convergence test
        if verbose and (tol == 0 or n_iter % 10 != 0):
            end_time = time.time()
            print("Epoch %02d reached after %.3f seconds." %
                  (n_iter, end_time - start_time))
        return W, H, n_iter
            
    if solver == 'cd':
        # so W and Ht are both in C order in memory
        Ht = check_array(H.T, order='C')
        X = check_array(X, accept_sparse='csr')
    
        rng = check_random_state(random_state)
    
        for n_iter in range(max_iter):
            violation = 0.
    
            # Update W
            violation += _update_coordinate_descent(X, W, Ht, shuffle, rng)
            # Update H
            if update_H:
                violation += _update_coordinate_descent(X.T, Ht, W, shuffle, rng)
    
            if n_iter == 0:
                violation_init = violation
    
            if violation_init == 0:
                break
    
            if verbose:
                print("violation:", violation / violation_init)
    
            if violation / violation_init <= tol:
                if verbose:
                    print("Converged at iteration", n_iter + 1)
                break
            
        return W, Ht.T, n_iter
        

def CoxNMF(X: np.ndarray,
           t: np.ndarray,
           e: np.ndarray,
           W_init = None,
           H_init = None,
           n_components: Optional[int] = 10,
           alpha: Optional[float] = 1e-5,
           sigma: Optional[float] = 0,
           penalizer: Optional[float] = 0,
           l1_ratio: Optional[float] = 0,
           ci_tol: Optional[float] = 0.02,
           max_iter: Optional[int] = 1000,
           solver: Optional[str] = 'mu',
           update_rule: Optional[str] = 'projection',
           tol: Optional[float] = 1e-6,
           random_state: Optional[int] = None,
           update_H: bool = True,
           update_beta: bool = True,
           W_normalization: bool = False,
           H_normalization: bool = False,
           beta_normalization: bool = True,
           logger=None,
           verbose: Optional[int] = 0):
    '''
    Parameters
    ----------
    X     : array-like, shape (n_samples, n_features)
            Constant input matrix.
    W     : array-like, shape (n_samples, n_components)
            Initial guess for the solution.
    H     : array-like, shape (n_components, n_features)
            Initial guess for the solution.
        
    t     : array-like, shape (n_components)
            Survival time.
        
    e     : array-like, shape (n_components)
            Survival event (death = 1).
        
    alpha : scalar value.
            parameter used for learning the H guided by Cox model.
            
    ci_tol: Tolerace of decrease of oncordance index to stop iteration.
    '''
    
    if W_init is None or H_init is None:
        W, H = _initialize_nmf(X, n_components, init = 'random', random_state=random_state)
    else:
        W, H = W_init, H_init
        
    # used for the convergence criterion
    error_at_init = calcuate_Frobenius_norm(X, W, H, square_root=True)
    previous_error = error_at_init
    max_cindex = 0.5
    start_time = time.time()
    HHt, XHt = None, None
    t_geq_matrix = np.array([[int(y >= x) for i,x in enumerate(t)] for j,y in enumerate(t)])
    error_list = []
    cindex_list = []
    max_cindex_res = None
    beta = None
    
    for n_iter in range(1, max_iter + 1):
        # update W
        # HHt and XHt are saved and reused if not update_H
        if sigma == 0:
            delta_W, HHt, XHt = _multiplicative_update_w(X, W, H, HHt, XHt, update_H=update_H)
        elif sigma > 0:
            delta_W, HHt, XHt = _multiplicative_update_w_orth(X, W, H, HHt, XHt, sigma = sigma)
            
        W *= delta_W
        if W_normalization:
            # column normalization on W
            W = (W / np.linalg.norm(W, axis=0).T)
    
        if update_beta:
            
            beta, ll_, hessian = newton_rhapson_for_efron_model(X=H.T,
                                                                T=t,
                                                                E=e,
                                                                initial_point=beta,
                                                                penalizer=penalizer,
                                                                l1_ratio=l1_ratio,
                                                                max_steps=1)
            # normalize beta
            if beta_normalization:
                beta = beta / (np.max(beta)-np.min(beta))
            cindex = concordance_index(t, -np.dot(H.T, beta), e)
            
        # update H
        if update_H:
            n_patients = t.shape[0]
            numerator = safe_sparse_dot(W.T, X)
            denominator = np.dot(np.dot(W.T, W), H)
            H_mu = H*(numerator/denominator)
            
            if beta is not None:
                cox_numerator = np.repeat(np.expand_dims(np.matmul(beta, np.exp(np.matmul(beta.T, H)) ), axis = 2), n_patients, axis = 2).swapaxes(1,2) * t_geq_matrix.T
                cox_numerator[:, np.where(e==0)[0], :] = 0
                cox_denominator = np.expand_dims(np.matmul(np.exp(np.matmul(beta.T, H)), t_geq_matrix), axis = 2)
                cox_fraction = e * np.repeat(beta, n_patients, axis = 1) - np.sum(cox_numerator / cox_denominator, axis = 1)
    
                H_partial = alpha / 2 * (numerator/denominator) * cox_fraction
                
                
                if update_rule == 'projection':
                    H_partial[H_partial < 0] = 0
                
                H = H_mu + H_partial
            else:
                H = H_mu
            
            if np.sum(np.isnan(H)) > 0:
                print('Detected NaN value in CoxNMF @H. Possibly due to overflow large value in exp(beta*H). Algorithm stopped. H row normalization is suggested.')
                break
            
            if H_normalization:
                # row normalization on H
                H = (H.T / np.linalg.norm(H, axis=1)).T
            
            # These values will be recomputed since H changed
            HHt, XHt = None, None
    
        error = calcuate_Frobenius_norm(X, W, H, square_root=True)
        relative_error = error/np.linalg.norm(X,'fro')
        if verbose:
            print("Epoch %04d error: %f, relative_error: %f, concordance index: %f" % (n_iter, error, relative_error, cindex))
        if logger:
            logger.log(logging.INFO, "Epoch %04d error: %f, relative_error: %f, concordance index: %f" % (n_iter, error, relative_error, cindex))
        
        error_list.append(error)
        cindex_list.append(cindex)
        
        # test convergence criterion every 10 iterations
#        if tol > 0 and n_iter % 10 == 0:
        if n_iter % 10 == 0:
            if (previous_error - error) / error_at_init < tol:
                print('Detected non-decreasing NMF error. Algorithm stopped.')
                break
            previous_error = error
            
            if (cindex - max_cindex) < - ci_tol: # if new concordance index smaller than previous 0.02
                print('Detected non-increasing C-Index. Algorithm stopped.')
                break
        
        if cindex >= max_cindex:
            max_cindex = cindex
            max_cindex_res = {}
            max_cindex_res['W'] = W
            max_cindex_res['H'] = H
            max_cindex_res['error'] = error
            max_cindex_res['cindex'] = cindex
            max_cindex_res['beta'] = beta.reshape(-1)
            
        
    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %04d reached after %.3f seconds." %
              (n_iter, end_time - start_time))
    return W, H, n_iter, error_list, cindex_list, max_cindex_res




