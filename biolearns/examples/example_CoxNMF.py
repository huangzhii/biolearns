# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Tue Aug 11 13:25:15 2020
# Author: Zhi Huang, Purdue University

import sys, os
import numpy as np
import pandas as pd

from biolearns import TCGA
from biolearns import CoxNMF
from biolearns import expression_filter

alpha = 1e-2
cancer = 'BRCA'
data = TCGA(cancer)
bcd_m = [b[:12] for b in data.mRNAseq.columns]
bcd_p = [b[:12] for b in data.clinical.index]
bcd = np.intersect1d(bcd_m, bcd_p)

clinical = data.clinical
mRNAseq = data.mRNAseq.iloc[:,np.nonzero(np.in1d(bcd, bcd_m))[0]]

X = expression_filter(mRNAseq, 0.2, 0.2).values
t = data.overall_survival_time[np.nonzero(np.in1d(bcd, bcd_p))[0]]
e = data.overall_survival_event[np.nonzero(np.in1d(bcd, bcd_p))[0]]
# remove NaN
list2keep = [~(nant or nane) and ti >= 0 for nant, nane, ti in zip(np.isnan(t), np.isnan(e), t)]
if cancer == 'BRCA':
    #remove male patients
    list2keep = list2keep & (clinical.gender.loc[bcd] == 'female').values

X = X[:, list2keep]
t = t[list2keep]
e = e[list2keep]

X = np.log2(X+1)
np.random.seed(11)
W, H, n_iter, error_list, cindex_list, max_cindex_res = \
                            CoxNMF(X,t,e,
                                   n_components=120,
                                   max_iter = 1000,
                                   penalizer = 1e-8,
                                   ci_tol=0.2,
                                   alpha=alpha,
                                   l1_ratio=1,
                                   H_row_normalization = False,
                                   verbose=1)


# from lifelines import CoxPHFitter
# cph = CoxPHFitter()
# data = pd.DataFrame(X.T)
# data['t'] = t
# data['e'] = e
# cph.fit(data, 't', 'e')

    
