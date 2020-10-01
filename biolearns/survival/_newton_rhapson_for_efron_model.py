#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:30:55 2020

@author: zhihuan

The enclosed functions are rewrited from python package "lifelines" with specific
version=0.25.2.
lifelines is one of the greatest python packages for survival analysis, it provides
many useful functions. However, the package itself updates  frequently and may not
always compatible with biolearns. Thus we extract these lifelines functions in the
version free manner.

"""

from typing import Callable, Iterator, List, Optional, Tuple, Union, Any, Iterable
from scipy.linalg import solve as spsolve, norm
import warnings

from numpy import dot, einsum, log, exp, zeros, arange, multiply, ndarray
import numpy as np
import pandas as pd
from autograd import elementwise_grad
# from lifelines.utils import StepSizer
import time
from autograd import numpy as anp
from lifelines.utils import concordance_index

def _get_efron_values_single(
    X: pd.DataFrame,
    T: pd.Series,
    E: pd.Series,
    weights: pd.Series,
    entries: None,
    beta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates the first and second order vector differentials, with respect to beta.
    Note that X, T, E are assumed to be sorted on T!

    A good explanation for Efron. Consider three of five subjects who fail at the time.
    As it is not known a priori that who is the first to fail, so one-third of
    (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
    of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.

    From https://cran.r-project.org/web/packages/survival/survival.pdf:

    "Setting all weights to 2 for instance will give the same coefficient estimate but halve the variance. When
    the Efron approximation for ties (default) is employed replication of the data will not give exactly the same coefficients as the
    weights option, and in this case the weighted fit is arguably the correct one."

    Parameters
    ----------
    X: array
        (n,d) numpy array of observations.
    T: array
        (n) numpy array representing observed durations.
    E: array
        (n) numpy array representing death events.
    weights: array
        (n) an array representing weights per observation.
    beta: array
        (1, d) numpy array of coefficients.

    Returns
    -------
    hessian:
        (d, d) numpy array,
    gradient:
        (1, d) numpy array
    log_likelihood: float
    """

    X = X.values
    T = T.values
    E = E.values
    weights = weights.values

    n, d = X.shape # n: samples; d: variables
    hessian = zeros((d, d))
    gradient = zeros((d,))
    log_lik = 0

    # Init risk and tie sums to zero
    x_death_sum = zeros((d,))
    risk_phi, tie_phi = 0, 0
    risk_phi_x, tie_phi_x = zeros((d,)), zeros((d,))
    risk_phi_x_x, tie_phi_x_x = zeros((d, d)), zeros((d, d))

    # Init number of ties and weights
    weight_count = 0.0
    tied_death_counts = 0
    scores = weights * exp(dot(X, beta))

    phi_x_is = scores[:, None] * X
    phi_x_x_i = np.empty((d, d))

    # Iterate backwards to utilize recursive relationship
    for i in range(n - 1, -1, -1): # i = n-1, n-2, n-3, ..., 3, 2, 1, 0
        # Doing it like this to preserve shape
        ti = T[i]
        ei = E[i]
        xi = X[i]
        w = weights[i]

        # Calculate phi values
        phi_i = scores[i]
        phi_x_i = phi_x_is[i]
        # https://stackoverflow.com/a/51481295/1895939
        phi_x_x_i = multiply.outer(xi, phi_x_i)

        # Calculate sums of Risk set
        risk_phi = risk_phi + phi_i
        risk_phi_x = risk_phi_x + phi_x_i
        risk_phi_x_x = risk_phi_x_x + phi_x_x_i

        # Calculate sums of Ties, if this is an event
        if ei:
            x_death_sum = x_death_sum + w * xi
            tie_phi = tie_phi + phi_i
            tie_phi_x = tie_phi_x + phi_x_i
            tie_phi_x_x = tie_phi_x_x + phi_x_x_i

            # Keep track of count
            tied_death_counts += 1
            weight_count += w

        if i > 0 and T[i - 1] == ti:
            # There are more ties/members of the risk set
            continue
        elif tied_death_counts == 0:
            # Only censored with current time, move on
            continue

        # There was at least one event and no more ties remain. Time to sum.
        # This code is near identical to the _batch algorithm below. In fact, see _batch for comments.
        weighted_average = weight_count / tied_death_counts

        if tied_death_counts > 1:
            increasing_proportion = arange(tied_death_counts) / tied_death_counts
            denom = 1.0 / (risk_phi - increasing_proportion * tie_phi)
            numer = risk_phi_x - multiply.outer(increasing_proportion, tie_phi_x)
            a1 = einsum("ab,i->ab", risk_phi_x_x, denom) - einsum("ab,i->ab", tie_phi_x_x, increasing_proportion * denom)
        else:
            denom = 1.0 / np.array([risk_phi])
            numer = risk_phi_x
            a1 = risk_phi_x_x * denom

        summand = numer * denom[:, None]
        a2 = summand.T.dot(summand)

        gradient = gradient + x_death_sum - weighted_average * summand.sum(0)

        log_lik = log_lik + dot(x_death_sum, beta) + weighted_average * log(denom).sum()
        hessian = hessian + weighted_average * (a2 - a1)

        # reset tie values
        tied_death_counts = 0
        weight_count = 0.0
        x_death_sum = zeros((d,))
        tie_phi = 0
        tie_phi_x = zeros((d,))
        tie_phi_x_x = zeros((d, d))

    return hessian, gradient, log_lik




def newton_rhapson_for_efron_model(
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    weights: Optional[pd.Series] = None,
    entries: Optional[pd.Series] = None,
    initial_point: Optional[np.ndarray] = None,
    step_size: Optional[float] = None,
    l1_ratio: Optional[float] = 1,
    penalizer: Optional[float] = None,
    precision: float = 1e-07,
    show_progress: bool = False,
    max_steps: int = 500
):  # pylint: disable=too-many-statements,too-many-branches
    """
    Newton Rhaphson algorithm for fitting CPH model.
    Note
    ----
    The data is assumed to be sorted on T!
    Parameters
    ----------
    X: (n,d) Pandas DataFrame of observations.
    T: (n) Pandas Series representing observed durations.
    E: (n) Pandas Series representing death events.
    weights: (n) an iterable representing weights per observation.
    initial_point: (d,) numpy array of initial starting point for
                  NR algorithm. Default 0.
    step_size: float, optional
        > 0.001 to determine a starting step size in NR algorithm.
    precision: float, optional
        the convergence halts if the norm of delta between
        successive positions is less than epsilon.
    show_progress: bool, optional
        since the fitter is iterative, show convergence
             diagnostics.
    max_steps: int, optional
        the maximum number of iterations of the Newton-Rhaphson algorithm.
    Returns
    -------
    beta: (1,d) numpy array.
    """
    CONVERGENCE_DOCS = "Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model"
    
    
    n, d = X.shape # n: samples, d: variables
    
    idx = sorted(range(len(T)), key=T.__getitem__) # order_ascending
    ridx = sorted(range(len(T)), key=idx.__getitem__)

    X = X[idx,:]
    T = T[idx]
    E = E[idx]
    
    
    X = pd.DataFrame(X)
    T = pd.Series(T)
    E = pd.Series(E)
    
    
    if not weights:
        weights = pd.Series([1]*n)

    
    if penalizer:
        # soft penalizer functions, from https://www.cs.ubc.ca/cgi-bin/tr/2009/TR-2009-19.pdf
        soft_abs = lambda x, a: 1 / a * (anp.logaddexp(0, -a * x) + anp.logaddexp(0, a * x))
        elastic_net_penalty = (
            lambda beta, a: n
            * 0.5
            * (
                l1_ratio * (penalizer * soft_abs(beta, a)).sum()
                + (1 - l1_ratio) * (penalizer * beta ** 2).sum()
            )
        )
        d_elastic_net_penalty = elementwise_grad(elastic_net_penalty)
        dd_elastic_net_penalty = elementwise_grad(d_elastic_net_penalty)

    # get_gradients = _choose_gradient_calculator(T, X, entries)

    # make sure betas are correct size.
    if initial_point is not None:
        beta = initial_point.reshape(-1)
    else:
        beta = np.zeros((d,))

    # step_sizer = StepSizer(step_size)
    # step_size = step_sizer.next()
    step_size = 1

    delta = np.zeros_like(beta)
    converging = True
    success = False
    beta_curr, hessian = delta, None
    ll_, previous_ll_ = 0.0, 0.0
    start = time.time()
    i = 0
    while converging:
        beta += step_size * delta

        i += 1
        h, g, ll_ = _get_efron_values_single(X, T, E, weights, entries, beta)
        
        if np.sum(np.isnan(h)) > 0:
            raise ValueError('NaN detected in Hessian or gradient. Possibly large value in X or in beta. Program stopped.')
        
        if penalizer:
            ll_ -= elastic_net_penalty(beta, 1e10)
            g -= d_elastic_net_penalty(beta, 1e10)
            h[np.diag_indices(d)] -= dd_elastic_net_penalty(beta, 1e10)

        # reusing a piece to make g * inv(h) * g.T faster later
        # try:
        # inv_h_dot_g_T = spsolve(-h, g, assume_a='pos', check_finite=False)
        try:
            inv_h_dot_g_T = np.linalg.solve(-h,g)
        except:
            print('numpy.linalg.LinAlgError: Singular matrix. Use least square instead.')
            inv_h_dot_g_T = np.linalg.lstsq(-h,g)[0]
        delta = inv_h_dot_g_T
            
        beta_curr = (beta + step_size * delta).reshape(-1,1)
        
        hessian, gradient = h, g
    
        if delta.size > 0:
            norm_delta = norm(delta)
        else:
            norm_delta = 0

        # reusing an above piece to make g * inv(h) * g.T faster.
        newton_decrement = g.dot(inv_h_dot_g_T) / 2

    
        if show_progress:
            print(
                "\rIteration %d: norm_delta = %.5f, step_size = %.4f, log_lik = %.5f, newton_decrement = %.5f, seconds_since_start = %.1f"
                % (i, norm_delta, step_size, ll_, newton_decrement, time.time() - start)
            )

        # convergence criteria
        if norm_delta < precision:
            converging, success = False, True
        elif previous_ll_ != 0 and abs(ll_ - previous_ll_) / (-previous_ll_) < 1e-09:
            # this is what R uses by default
            converging, success = False, True
        elif newton_decrement < precision:
            converging, success = False, True
        elif i >= max_steps:
            # 50 iterations steps with N-R is a lot.
            # Expected convergence is ~10 steps
            converging, success = False, False
        elif abs(ll_) < 0.0001 and norm_delta > 1.0:
            warnings.warn("The log-likelihood is getting suspiciously close to 0 and the delta is still large. There may be complete separation in the dataset. This may result in incorrect inference of coefficients. \
                          See https://stats.stackexchange.com/q/11109/11867 for more.\n")
            converging, success = False, False

        previous_ll_ = ll_
        # step_size = step_sizer.update(norm_delta).next()

    if show_progress and success:
        print("Convergence success after %d iterations." % (i))
    elif show_progress and not success:
        print("Convergence failed. See any warning messages.")

    # report to the user problems that we detect.
    if success and norm_delta > 0.1:
        warnings.warn(
            "Newton-Rhaphson convergence completed successfully but norm(delta) is still high, %.3f. This may imply non-unique solutions to the maximum likelihood. Perhaps there is collinearity or complete separation in the dataset?\n"
            % norm_delta
        )
    elif not success:
        warnings.warn(
            "Newton-Rhaphson failed to converge sufficiently in %d steps.\n" % max_steps
        )

    return beta_curr, ll_, hessian
