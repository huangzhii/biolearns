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
import scipy.sparse as sp
import pandas as pd
from sklearn.utils import check_random_state, check_array
from sklearn.decomposition.nmf import _initialize_nmf
from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast
from sklearn.utils.extmath import safe_sparse_dot
import copy

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
        

def CoxNMF(X, t, e, n_components, alpha=1e-5, sigma = 0, eta_b = None, cph_penalizer=0, l1_ratio = 0, ci_tol=0.02, solver='mu', update_rule='projection', cph_max_steps=1, max_iter=1000, tol=1e-6, random_state=None, update_H=True, update_beta=True, H_row_normalization=False, logger=None, verbose=0):
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
            
    sigma : scalar value.
            orthogonal constraint on W.
            
    eta_b : scalar value.
            step size in Cox model.
            
    ci_tol: Tolerace of decrease of oncordance index to stop iteration.
    '''
    W, H = _initialize_nmf(X, n_components, init = 'random', random_state=random_state)
        
    # used for the convergence criterion
    error_at_init = calcuate_Frobenius_norm(X, W, H, square_root=True)
    previous_error = error_at_init
    max_cindex = 0.5
    start_time = time.time()
    HHt, XHt = None, None
    t_geq_matrix = np.array([[int(y >= x) for i,x in enumerate(t)] for j,y in enumerate(t)])
    error_list = []
    cindex_list = []
    for n_iter in range(1, max_iter + 1):
        # update W
        # HHt and XHt are saved and reused if not update_H
        if sigma == 0:
            delta_W, HHt, XHt = _multiplicative_update_w(X, W, H, HHt, XHt, update_H=update_H)
        elif sigma > 0:
            delta_W, HHt, XHt = _multiplicative_update_w_orth(X, W, H, HHt, XHt, sigma = sigma)
            
        W *= delta_W
#        if H_row_normalization:
#            W = (W.T / np.linalg.norm(W, axis=1).T).T
        
        beta, cph = None, None
        if update_beta:
            H_cox = pd.DataFrame(H.T)
            H_cox['time'] = t
            H_cox['event'] = e
            if cph_penalizer > 0:
                cph = StepCoxPHFitter(penalizer = cph_penalizer, l1_ratio = l1_ratio)
            else:
                cph = StepCoxPHFitter()
            cph.max_iterations = cph_max_steps
            if not beta:
                initial_point = None
            else:
                initial_point = beta.reshape(-1)
            cph.fit(H_cox, initial_point = initial_point, duration_col='time', event_col='event', step_size = eta_b, show_progress=False)
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
            if H_row_normalization:
                H = (H.T / np.linalg.norm(H, axis=1)).T
            
            # These values will be recomputed since H changed
            HHt, XHt = None, None
    
        error = calcuate_Frobenius_norm(X, W, H, square_root=True)
        if verbose:
            print("Epoch %04d error: %f, concordance index: %f" % (n_iter, error, cindex))
        if logger:
            logger.log(logging.INFO, "Epoch %04d error: %f, concordance index: %f" % (n_iter, error, cindex))
        
        error_list.append(error)
        cindex_list.append(cindex)
        
        # test convergence criterion every 10 iterations
#        if tol > 0 and n_iter % 10 == 0:
        if (previous_error - error) / error_at_init < tol:
            break
        previous_error = error
        if (cindex - max_cindex) < -ci_tol: # if new concordance index smaller than previous 0.02
            break
        
        if cindex >= max_cindex:
            max_cindex = cindex
            max_cindex_res = {}
            max_cindex_res['W'] = W
            max_cindex_res['H'] = H
            max_cindex_res['cph'] = copy.deepcopy(cph)
            max_cindex_res['error'] = error
            max_cindex_res['cindex'] = cindex
            
        
    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %04d reached after %.3f seconds." %
              (n_iter, end_time - start_time))
    return W, H, cph, n_iter, error_list, cindex_list, max_cindex_res




