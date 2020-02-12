#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:11:25 2020

@author: Zhi Huang
"""

from biolearns.dataset.TCGA import TCGACancer
from biolearns.coexpression.lmQCM import lmQCM
from biolearns.preprocessing.filter import expression_filter
from biolearns.survival import logranktest



brca = TCGACancer('KICH')
mRNAseq = brca.get_mRNAseq()
clinical = brca.get_clinical()

mRNAseq = expression_filter(mRNAseq, meanq = 0.5, varq = 0.5)
lobj = lmQCM(mRNAseq)
lobj.fit()

