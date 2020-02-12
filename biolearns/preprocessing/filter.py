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

def expression_filter(x, meanq = 0.2, varq = 0.2):
    '''
    Parameters
    ----------
    x     : Real-valued expression matrix with rownames indicating
            gene ID or gene symbol.
    meanq : By which genes with low expression mean across samples are filtered out.
    varq  : By which genes with low expression variance across samples are filtered out.
        
    Returns
    -------
    x     : Real-valued expression matrix with rownames indicating
            gene ID or gene symbol.
    '''
    mean_quantile = x.mean(axis = 1).quantile(q=meanq)
    x = x.loc[x.mean(axis = 1) >= mean_quantile,:]
    var_quantile = x.var(axis = 1).quantile(q=varq)
    x = x.loc[x.var(axis = 1) >= var_quantile,:]
    return x
