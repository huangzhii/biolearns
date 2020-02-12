#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:11:25 2020

@author: Zhi Huang
"""

import numpy as np
from biolearns.dataset.TCGA import TCGACancer
from biolearns.coexpression.lmQCM import lmQCM
from biolearns.preprocessing.filter import expression_filter
from biolearns.survival import logranktest

brca = TCGACancer('LAML')
mRNAseq = brca.mRNAseq
clinical = brca.clinical

# =============================================================================
# Gene Co-expression Analysis
# =============================================================================
mRNAseq = expression_filter(mRNAseq, meanq = 0.5, varq = 0.5)
lobj = lmQCM(mRNAseq)
lobj.fit()


# =============================================================================
# Survival Analysis
# =============================================================================
genes = mRNAseq.index.values.astype('str')
ratio = mRNAseq.loc['AAK1',].values

bcd_m = [b[:12] for b in mRNAseq.columns]
bcd_p = [b[:12] for b in clinical.index]
bcd = np.intersect1d(bcd_m, bcd_p)

ratio = ratio[np.nonzero(np.in1d(bcd_m, bcd))[0]]
t = brca.overall_survival_time[np.nonzero(np.in1d(bcd_p, bcd))[0]]
e = brca.overall_survival_event[np.nonzero(np.in1d(bcd_p, bcd))[0]]

logrank_results, fig = logranktest(ratio[~np.isnan(t)], t[~np.isnan(t)], e[~np.isnan(t)])

