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
import numpy as np
import scipy as sp
import pandas as pd
from scipy import sparse
from sklearn.utils import check_random_state, check_array
from sklearn.decomposition._nmf import _initialize_nmf
from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from lifelines.utils import concordance_index
from ..survival import StepCoxPHFitter
import time
import warnings
import logging
EPSILON = np.finfo(np.float32).eps

    
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

def safe_sparse_dot(a, b, dense_output=False):
    # this is from sklearn.utils.extmath
    """Dot product that handle the sparse matrix case correctly
    
    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, (default=False)
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sparse.issparse(a) and sparse.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret


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
        HHt, XHt = None, None, None
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
                HHt, XHt = None, None, None
        
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
        

def CoxNMF(X, t, e, n_components, alpha=1, solver='mu', update_rule='projection', cph_max_steps=1, max_iter=1000, tol=1e-6, random_state=None, update_H=True, update_beta=True, logger=None, verbose=0):
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
    '''
    W, H = _initialize_nmf(X, n_components, init = 'random', random_state=random_state)
        
    # used for the convergence criterion
    error_at_init = calcuate_Frobenius_norm(X, W, H, square_root=True)
    previous_error = error_at_init
    
    start_time = time.time()
    HHt, XHt = None, None, None
    t_geq_matrix = np.array([[int(y >= x) for i,x in enumerate(t)] for j,y in enumerate(t)])

    for n_iter in range(1, max_iter + 1):
        # update W
        # HHt and XHt are saved and reused if not update_H
        delta_W, HHt, XHt = _multiplicative_update_w(X, W, H, HHt, XHt, update_H = update_H)
        W *= delta_W
        
        beta = None
        if update_beta:
            H_cox = pd.DataFrame(H.T)
            H_cox['time'] = t
            H_cox['event'] = e
            cph = StepCoxPHFitter()
            cph.max_iterations = cph_max_steps
            cph.fit(H_cox, duration_col='time', event_col='event', show_progress=False)
            cindex = concordance_index(H_cox['time'], -cph.predict_partial_hazard(H.T), H_cox['event'])
            
            beta = cph.params_.values.reshape(-1,1)
        # update H
        if update_H:
            n_patients = t.shape[0]
            numerator = safe_sparse_dot(W.T, X)
            denominator = np.dot(np.dot(W.T, W), H)
            H_mu = H*(numerator/denominator)
            
            cox_numerator = np.repeat(np.expand_dims(np.matmul(beta, np.exp(np.matmul(beta.T, H)) ), axis = 2), n_patients, axis = 2).swapaxes(1,2) * t_geq_matrix.T
            cox_numerator[:, np.where(e==0)[0], :] = 0
            cox_denominator = np.expand_dims(np.matmul(np.exp(np.matmul(beta.T, H)), t_geq_matrix), axis = 2)
            cox_fraction = e * np.repeat(beta, n_patients, axis = 1) - np.sum(cox_numerator / cox_denominator, axis = 1)

            H_partial = alpha / 2 * (numerator/denominator) * cox_fraction
            
            if update_rule == 'projection':
                H_partial[H_partial < 0] = 0
            
            H = H_mu + H_partial
            
            # These values will be recomputed since H changed
            HHt, XHt = None, None, None
    
        error = calcuate_Frobenius_norm(X, W, H, square_root=True)
        loss_train = np.linalg.norm(X - np.matmul(W, H), ord='fro')
        if verbose:
            print("Epoch %04d error: %f = %f, concordance index: %f" % (n_iter, error, loss_train, cindex))
        if logger:
            logger.log(logging.INFO, "Epoch %04d error: %f = %f, concordance index: %f" % (n_iter, error, loss_train, cindex))
            
        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %04d reached after %.3f seconds." %
              (n_iter, end_time - start_time))
    return W, H, n_iter
