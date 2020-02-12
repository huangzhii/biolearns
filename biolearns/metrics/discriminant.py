# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Tue Feb 11 12:29:35 2020
# Author: Zhi Huang, Purdue University
#
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

def fisher_discriminant(H, label):
    '''
    Parameters
    ----------
    H     : Real-valued matrix with columns indicating samples.
    label : Class indices.
        
    Returns
    -------
    E_D   : Real scalar value indicating fisher discriminant.
    Notes
    -----
    This fisher discriminant is the equation (3 a,b) in
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3495075
    
    label is further sorted in ascending order, then apply its order to label and H.
    Otherwise denominator will be wrong.
    
    References
    ----------
    .. [1] Lee SY, Song HA, Amari SI. A new discriminant NMF algorithm and its
           application to the extraction of subtle emotional differences in speech.
           Cognitive neurodynamics. 2012 Dec 1;6(6):525-35.

    '''
    order = np.argsort(label)
    H = H[:,order]
    label = label[order]
    numerator, denominator = 0, 0
    mu_rkn = np.zeros((H.shape[0], 0))
    mu_r_all = 1/H.shape[1] * np.sum(H, axis = 1)
    for k in np.unique(label):
        N_k = np.sum(k == label)
        mu_rk_block = np.zeros((0, N_k))
        for r in range(H.shape[0]):
            mu_r = mu_r_all[r]
            mu_rk = 1/N_k * np.sum(H[r, k == label])
            mu_rk_block = np.concatenate((mu_rk_block, np.array([mu_rk] * N_k).reshape(1,N_k)), axis = 0)
            numerator += N_k * (mu_rk - mu_r) ** 2
        mu_rkn = np.concatenate((mu_rkn, mu_rk_block), axis = 1)
    denominator = np.sum((H - mu_rkn)**2)
    E_D = numerator / denominator
    return E_D