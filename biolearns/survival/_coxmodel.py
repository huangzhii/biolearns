# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Wed Feb 19 14:23:17 2020
# Author: Zhi Huang, Purdue University
#
# This is a Child Class inherited from lifelines.CoxPHFitter
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
from numpy import dot, einsum, log, exp, zeros, arange, multiply, ndarray
import pandas as pd
from pandas import DataFrame, Series, Index
from lifelines.utils import normalize, coalesce, CensoringType
from lifelines import CoxPHFitter
from numpy.linalg import inv
from datetime import datetime
from typing import Callable, Iterator, List, Optional, Tuple, Union, Any, Iterable
import warnings

class StepCoxPHFitter(CoxPHFitter):
    def __init__(self, tie_method="Efron", penalizer=0.0, l1_ratio = 0.0, strata=None, baseline_estimation_method="breslow"):
        super(CoxPHFitter, self).__init__()
        if penalizer < 0:
            raise ValueError("penalizer parameter must be >= 0.")
        if tie_method != "Efron":
            raise NotImplementedError("Only Efron is available at the moment.")
        self.baseline_estimation_method = baseline_estimation_method
        self.l1_ratio = l1_ratio
        self.tie_method = tie_method
        self.penalizer = penalizer
        self.strata = strata
        self.cph_max_steps = 50
 

    @CensoringType.right_censoring
    def fit(
        self,
        df,
        duration_col=None,
        event_col=None,
        show_progress=False,
        initial_point=None,
        strata=None,
        step_size=None,
        weights_col=None,
        cluster_col=None,
        robust=False,
        batch_mode=None,
    ):
        """
        Fit the Cox proportional hazard model to a dataset.
        Parameters
        ----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
            `event_col` (see below), covariates columns, and special columns (weights, strata).
            `duration_col` refers to
            the lifetimes of the subjects. `event_col` refers to whether
            the 'death' events was observed: 1 if observed, 0 else (censored).
        duration_col: string
            the name of the column in DataFrame that contains the subjects'
            lifetimes.
        event_col: string, optional
            the  name of thecolumn in DataFrame that contains the subjects' death
            observation. If left as None, assume all individuals are uncensored.
        weights_col: string, optional
            an optional column in the DataFrame, df, that denotes the weight per subject.
            This column is expelled and not used as a covariate, but as a weight in the
            final regression. Default weight is 1.
            This can be used for case-weights. For example, a weight of 2 means there were two subjects with
            identical observations.
            This can be used for sampling weights. In that case, use `robust=True` to get more accurate standard errors.
        show_progress: boolean, optional (default=False)
            since the fitter is iterative, show convergence
            diagnostics. Useful if convergence is failing.
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.
        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.
        step_size: float, optional
            set an initial step size for the fitting algorithm. Setting to 1.0 may improve performance, but could also hurt convergence.
        robust: boolean, optional (default=False)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
            ties, so if there are high number of ties, results may significantly differ. See
            "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078
        cluster_col: string, optional
            specifies what column has unique identifiers for clustering covariances. Using this forces the sandwich estimator (robust variance estimator) to
            be used.
        batch_mode: bool, optional
            enabling batch_mode can be faster for datasets with a large number of ties. If left as None, lifelines will choose the best option.
        Returns
        -------
        self: CoxPHFitter
            self with additional new properties: ``print_summary``, ``hazards_``, ``confidence_intervals_``, ``baseline_survival_``, etc.
        Note
        ----
        Tied survival times are handled using Efron's tie-method.
        Examples
        --------
        >>> from lifelines import CoxPHFitter
        >>>
        >>> df = pd.DataFrame({
        >>>     'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        >>>     'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        >>>     'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>> })
        >>>
        >>> cph = CoxPHFitter()
        >>> cph.fit(df, 'T', 'E')
        >>> cph.print_summary()
        >>> cph.predict_median(df)
        >>> from lifelines import CoxPHFitter
        >>>
        >>> df = pd.DataFrame({
        >>>     'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        >>>     'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
        >>>     'weights': [1.1, 0.5, 2.0, 1.6, 1.2, 4.3, 1.4, 4.5, 3.0, 3.2, 0.4, 6.2],
        >>>     'month': [10, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>>     'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
        >>> })
        >>>
        >>> cph = CoxPHFitter()
        >>> cph.fit(df, 'T', 'E', strata=['month', 'age'], robust=True, weights_col='weights')
        >>> cph.print_summary()
        >>> cph.predict_median(df)
        """
        if duration_col is None:
            raise TypeError("duration_col cannot be None.")

        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"
        self.duration_col = duration_col
        self.event_col = event_col
        self.robust = robust
        self.cluster_col = cluster_col
        self.weights_col = weights_col
        self._n_examples = df.shape[0]
        self._batch_mode = batch_mode
        self.strata = coalesce(strata, self.strata)

        X, T, E, weights, original_index, self._clusters = self._preprocess_dataframe(df)

        self.durations = T.copy()
        self.event_observed = E.copy()
        self.weights = weights.copy()

        if self.strata is not None:
            self.durations.index = original_index
            self.event_observed.index = original_index
            self.weights.index = original_index

        self._norm_mean = X.mean(0)
        self._norm_std = X.std(0)

        # this is surprisingly faster to do...
        X_norm = pd.DataFrame(
            normalize(X.values, self._norm_mean.values, self._norm_std.values), index=X.index, columns=X.columns
        )

        params_, ll_, variance_matrix_, baseline_hazard_, baseline_cumulative_hazard_ = self._fit_model(
            X_norm, T, E, weights=weights, initial_point=initial_point, show_progress=show_progress, step_size=step_size, max_steps = self.cph_max_steps
        )

        self.log_likelihood_ = ll_
        self.variance_matrix_ = variance_matrix_
        self.params_ = pd.Series(params_, index=X.columns, name="coef")
        self.baseline_hazard_ = baseline_hazard_
        self.baseline_cumulative_hazard_ = baseline_cumulative_hazard_

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._predicted_partial_hazards_ = (
                self.predict_partial_hazard(X)
                .to_frame(name="P")
                .assign(T=self.durations.values, E=self.event_observed.values, W=self.weights.values)
                .set_index(X.index)
            )

        self.standard_errors_ = self._compute_standard_errors(X_norm, T, E, weights)
        self.confidence_intervals_ = self._compute_confidence_intervals()
        self.baseline_survival_ = self._compute_baseline_survival()

        if hasattr(self, "_concordance_index_"):
            del self._concordance_index_

        return self
    
    
    
    def _fit_model_breslow(
        self,
        X: DataFrame,
        T: Series,
        E: Series,
        weights: Series,
        initial_point: Optional[ndarray] = None,
        step_size: Optional[float] = None,
        show_progress: bool = True,
        max_steps: int = 1
    ):
        beta_, ll_, hessian_ = self._newton_rhapson_for_efron_model(
            X, T, E, weights, initial_point=initial_point, step_size=step_size, show_progress=show_progress
        )

        # compute the baseline hazard here.
        predicted_partial_hazards_ = (
            pd.DataFrame(np.exp(dot(X, beta_)), columns=["P"])
            .assign(T=T.values, E=E.values, W=weights.values)
            .set_index(X.index)
        )
        baseline_hazard_ = self._compute_baseline_hazards(predicted_partial_hazards_)
        baseline_cumulative_hazard_ = self._compute_baseline_cumulative_hazard(baseline_hazard_)

        # rescale parameters back to original scale.
        params_ = beta_ / self._norm_std.values
        if hessian_.size > 0:
            variance_matrix_ = pd.DataFrame(
                -inv(hessian_) / np.outer(self._norm_std, self._norm_std), index=X.columns, columns=X.columns
            )
        else:
            variance_matrix_ = pd.DataFrame(index=X.columns, columns=X.columns)

        return params_, ll_, variance_matrix_, baseline_hazard_, baseline_cumulative_hazard_
