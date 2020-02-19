#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:11:25 2020

@author: Zhi Huang
"""

import numpy as np
from biolearns.dataset import TCGA
from biolearns.coexpression import lmQCM
from biolearns.preprocessing.filter import expression_filter
from biolearns.survival import logranktest

brca = TCGA('BRCA')
mRNAseq = brca.mRNAseq
clinical = brca.clinical

# =============================================================================
# Gene Co-expression Analysis
# =============================================================================
mRNAseq = expression_filter(mRNAseq, meanq = 0.5, varq = 0.5)
lobj = lmQCM(mRNAseq)
clusters, genes, eigengene_mat = lobj.fit()



# =============================================================================
# Survival Analysis
# =============================================================================

r = mRNAseq.loc['ABLIM3',].values

bcd_m = [b[:12] for b in mRNAseq.columns]
bcd_p = [b[:12] for b in clinical.index]
bcd = np.intersect1d(bcd_m, bcd_p)

r = r[np.nonzero(np.in1d(bcd, bcd_m))[0]]
t = brca.overall_survival_time[np.nonzero(np.in1d(bcd, bcd_p))[0]]
e = brca.overall_survival_event[np.nonzero(np.in1d(bcd, bcd_p))[0]]

logrank_results, fig = logranktest(r[~np.isnan(t)], t[~np.isnan(t)], e[~np.isnan(t)])

test_statistic, p_value = logrank_results.test_statistic, logrank_results.p_value
    
