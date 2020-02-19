#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:11:25 2020

@author: Zhi Huang
"""

import numpy as np
from biolearns.dataset import TCGA
from biolearns.preprocessing import expression_filter

brca = TCGA('BRCA')
mRNAseq = brca.mRNAseq
clinical = brca.clinical
mRNAseq = expression_filter(mRNAseq, meanq = 0.5, varq = 0.5)

# =============================================================================
# Gene Co-expression Analysis
# =============================================================================
from biolearns.coexpression import lmQCM

lobj = lmQCM(mRNAseq)
clusters, genes, eigengene_mat = lobj.fit()



# =============================================================================
# Survival Analysis
# =============================================================================
from biolearns.survival import logranktest

r = mRNAseq.loc['ABLIM3',].values

bcd_m = [b[:12] for b in mRNAseq.columns]
bcd_p = [b[:12] for b in clinical.index]
bcd = np.intersect1d(bcd_m, bcd_p)

r = r[np.nonzero(np.in1d(bcd, bcd_m))[0]]
t = brca.overall_survival_time[np.nonzero(np.in1d(bcd, bcd_p))[0]]
e = brca.overall_survival_event[np.nonzero(np.in1d(bcd, bcd_p))[0]]

logrank_results, fig = logranktest(r[~np.isnan(t)], t[~np.isnan(t)], e[~np.isnan(t)])

test_statistic, p_value = logrank_results.test_statistic, logrank_results.p_value
    


# =============================================================================
# CoxNMF
# =============================================================================
from biolearns.decomposition import CoxNMF

bcd_m = [b[:12] for b in mRNAseq.columns]
bcd_p = [b[:12] for b in clinical.index]
bcd = np.intersect1d(bcd_m, bcd_p)

X = mRNAseq.values[:,np.nonzero(np.in1d(bcd, bcd_m))[0]]
t = brca.overall_survival_time[np.nonzero(np.in1d(bcd, bcd_p))[0]]
e = brca.overall_survival_event[np.nonzero(np.in1d(bcd, bcd_p))[0]]
n_components = 16

CoxNMF(X, n_components=16, t=t, e=e)
