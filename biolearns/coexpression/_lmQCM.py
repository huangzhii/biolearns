# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Mon Feb 10 17:57:08 2020
# Author: Zhi Huang, Purdue University
#      ___       ___           ___           ___           ___     
#     /\__\     /\__\         /\  \         /\  \         /\__\    
#    /:/  /    /::|  |       /::\  \       /::\  \       /::|  |   
#   /:/  /    /:|:|  |      /:/\:\  \     /:/\:\  \     /:|:|  |   
#  /:/  /    /:/|:|__|__    \:\~\:\  \   /:/  \:\  \   /:/|:|__|__ 
# /:/__/    /:/ |::::\__\    \:\ \:\__\ /:/__/ \:\__\ /:/ |::::\__\
# \:\  \    \/__/~~/:/  /     \:\/:/  / \:\  \  \/__/ \/__/~~/:/  /
#  \:\  \         /:/  /       \::/  /   \:\  \             /:/  / 
#   \:\  \       /:/  /        /:/  /     \:\  \           /:/  /  
#    \:\__\     /:/  /        /:/  /       \:\__\         /:/  /   
#     \/__/     \/__/         \/__/         \/__/         \/__/    
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


from typing import Tuple, List, Optional
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

class lmQCM():
    '''
    Parameters
    ----------
        
    data_in        : real-valued expression matrix with rownames indicating
                     gene ID or gene symbol.
    gamma          : gamma value (default = 0.55)
    t              : t value (default = 1)
    lambda         : lambda value (default = 1)
    beta           : beta value (default = 0.4)
    minClusterSize : minimum length of cluster to retain (default = 10)
    CCmethod       : Methods for correlation coefficient calculation (default =
                     "pearson"). Users can also pick "spearman".
    normalization  : Determine if normalization is needed on massive correlation
                     coefficient matrix.
    Returns
    -------
    None
    Notes
    -----
    References
    ----------
    .. [1] Zhang J, Huang K. Normalized lmqcm: An algorithm for detecting weak quasi-cliques
           in weighted graph with applications in gene co-expression module discovery in
           cancers. Cancer informatics. 2014 Jan;13:CIN-S14021.
    .. [2] Huang Z, Han Z, Wang T, Shao W, Xiang S, Salama P, Rizkalla M, Huang K, Zhang J.
           TSUNAMI: Translational Bioinformatics Tool Suite For Network Analysis And Mining.
           bioRxiv. 2019 Jan 1:787507.
    Examples
    -------
    >>> tcga_COAD_data = 'http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/COAD/20160128/gdac.broadinstitute.org_COAD.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz'
    >>> data_in = pd.read_csv(tcga_COAD_data, header=0, skiprows=range(1, 2), index_col=0, sep='\t')
    >>> lobject = lmQCM(data_in)
    >>> lobject.fit()
    >>> lobject.clusters
    >>> lobject.clusters_names
    >>> lobject.eigengene_matrix
    '''
    def __init__(self,
                 data_in: pd.DataFrame,
                 gamma: Optional[float] = 0.55,
                 t: Optional[float] = 1,
                 lambdaa: Optional[float] = 1,
                 beta: Optional[float] = 0.4,
                 minClusterSize: Optional[int] = 10,
                 CCmethod: Optional[str] = "pearson",
                 normalization: Optional[bool] = False,
                 positive_corr: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        
        super(lmQCM, self).__init__(**kwargs)
        self.data_in = data_in
        if not isinstance(self.data_in, pd.DataFrame):
            print('Input matrix is not pandas DataFrame. Convert it to pandas.core.frame.DataFrame...')
            self.data_in = pd.DataFrame(self.data_in)
        if np.sum(np.isnan(self.data_in.values)) > 0:
            warnings.warn('%d NaN value detected. Replacing them to zero...' % np.sum(np.isnan(self.data_in.values)))
            self.data_in.fillna(0, inplace = True)
        self.gamma = gamma
        self.t = t
        self.lambdaa = lambdaa
        self.beta = beta
        self.minClusterSize = minClusterSize
        self.CCmethod = CCmethod.lower()
        self.normalization = normalization
        self.positive_corr = positive_corr
        self._calculate_correlation_matrix()
        print('Initialization Done.')
    
    def _localMaximumQCM(self
                         ) -> List[List]:
        '''
        fit the lmQCM model.
        
        Returns
        -------
        C : list of lists
        '''
        C = []
        nRow = self.corr_mat.shape[0]
        maxV = np.max(self.corr_mat, axis = 0)
        maxInd = np.argmax(self.corr_mat, axis = 1)
        lm_ind = np.where(maxV == np.max(self.corr_mat[maxInd,], axis = 1))[0]
        maxEdges = np.stack((maxInd[lm_ind], lm_ind)).T
        maxW = maxV[lm_ind]
        # sortMaxV = np.sort(maxW, kind='mergesort')[::-1] # decreasing
        # sortMaxInd = np.argsort(maxW, kind='mergesort')[::-1]
        sortMaxV = -np.sort(-maxW, kind='mergesort')
        sortMaxInd = np.argsort(-maxW, kind='mergesort')
        sortMaxEdges = maxEdges[sortMaxInd, ]
        print("Number of Maximum Edges: %d" % len(sortMaxInd))
        currentInit = 0 # In R, the index start at 1.
        noNewInit = 0
        
        pbar = tqdm(total=len(sortMaxInd))
        nodesInCluster = []
        while (currentInit+1) <= len(sortMaxInd) and noNewInit == 0:
            pbar.update(1)
            if sortMaxV[currentInit] < (self.gamma * sortMaxV[1]):
                noNewInit = 1
            else:
                if sortMaxEdges[currentInit, 0] not in nodesInCluster and sortMaxEdges[currentInit, 1] not in nodesInCluster:
                    newCluster = list(sortMaxEdges[currentInit, ])
                    addingMode = 1
                    currentDensity = sortMaxV[currentInit]
                    nCp = 2
                    totalInd = np.arange(nRow)
                    remainInd = np.setdiff1d(totalInd, newCluster)
                    while addingMode == 1:
                        neighborWeights = np.sum(self.corr_mat[newCluster,:][:,remainInd], axis = 0) #neighborWeights = np.round(neighborWeights, 6) this part is inconsistent with R
                        maxNeighborWeight = max(neighborWeights)
                        maxNeighborInd = np.argmax(neighborWeights) # this part is inconsistent with R
                        c_v = maxNeighborWeight/nCp
                        alphaN = 1 - 1/(2 * self.lambdaa * (nCp + self.t))
                        if c_v >= alphaN * currentDensity:
                            newCluster = newCluster + [remainInd[maxNeighborInd]]
                            nCp = nCp + 1
                            currentDensity = (currentDensity * ((nCp - 1) * (nCp - 2)/2) + maxNeighborWeight)/(nCp * (nCp - 1)/2)
                            remainInd = np.setdiff1d(remainInd, remainInd[maxNeighborInd])
                        else:
                            addingMode = 0
                    nodesInCluster = nodesInCluster + newCluster
                    C = C + [newCluster]
            currentInit += 1
        print(" Calculation Finished.")
        pbar.close()
        return(C)
        
    def _merging_lmQCM(self,
                       C: List[List]
                       ) -> List[List]:
        '''
        fit the lmQCM model.
                
        Parameters
        ----------
        C             : list of lists.
        
        Returns
        ----------
        mergedCluster : list of lists
        '''
        print(" %d Modules before merging." % len(C))
        sizeC = [len(i) for i in C]
        # sortInd = np.argsort(sizeC, kind='mergesort')[::-1]
        sortInd = np.argsort(-np.array(sizeC), kind='mergesort')
        mergedCluster = [C[i] for i in sortInd if len(C[i]) >= self.minClusterSize]
        mergeOccur = 1
        currentInd = -1
        while mergeOccur == 1:
            mergeOccur = 0
            while currentInd < len(mergedCluster):
                currentInd += 1
                if currentInd < len(mergedCluster):
                    keepInd = list(np.arange(0,currentInd+1))
                    for j in np.arange(currentInd+1, len(mergedCluster)):
                        interCluster = np.intersect1d(mergedCluster[currentInd], mergedCluster[j])
                        if len(interCluster) >= self.beta * min(len(mergedCluster[j]), len(mergedCluster[currentInd])):
                            mergedCluster[currentInd] = list(np.union1d(mergedCluster[currentInd], mergedCluster[j]))
                            mergeOccur = 1
                        else:
                            keepInd += [j]
                    mergedCluster = [mergedCluster[i] for i in keepInd]
                    
            sizeMergedCluster = [len(mergedCluster[i]) for i in range(len(mergedCluster))]
            # sortMergedInd = np.argsort(sizeMergedCluster, kind='mergesort')[::-1]
            sortMergedInd = np.argsort(-np.array(sizeMergedCluster), kind='mergesort')
            mergedCluster = [mergedCluster[i] for i in sortMergedInd]
            currentInd = -1
        print(" %d Modules remain after merging." % len(mergedCluster))
        return mergedCluster
        
    def _calculate_correlation_matrix(self
                                     ) -> None:
        print("Calculating massive correlation coefficient ...")
        if self.CCmethod.lower() == "pearson":
            self.corr_mat = np.corrcoef(self.data_in.values)
            
            # # Rpython
            # import rpy2
            # print(rpy2.__version__)
            # import rpy2.robjects as ro
            # import rpy2.robjects.numpy2ri as n2r
            # n2r.activate()
            # from rpy2.robjects.conversion import localconverter
            # from rpy2.robjects import pandas2ri
            
            # r = ro.r
            # r.assign("data_in", self.data_in.values)
            # r('cMatrix <- cor(t(data_in))')
            
            # with localconverter(ro.default_converter + pandas2ri.converter):
            #     self.corr_mat = r("cMatrix")
            # # self.corr_mat = np.round(self.corr_mat,2)
            
        if self.CCmethod.lower() == "spearman": self.corr_mat = spearmanr(self.data_in.values.T).correlation
        np.fill_diagonal(self.corr_mat, 0)
        if self.positive_corr: # if use positive_corr, then ignore negative correlations.
            self.corr_mat = np.abs(self.corr_mat)
        
        if np.sum(np.isnan(self.corr_mat)) > 0:
            warnings.warn('%d NaN value detected in correlation matrix. Replacing them to zero...' % np.sum(np.isnan(self.corr_mat)))
            self.corr_mat[np.isnan(self.corr_mat)] = 0
        if self.normalization:
            D = np.sum(self.corr_mat, axis = 0)
            D_half = 1.0/np.sqrt(D)
            self.corr_mat = np.multiply(np.multiply(self.corr_mat, D_half).T, D_half)
        
    def fit(self
            ) -> Tuple[List[List], List, pd.DataFrame]:
        '''
        fit the lmQCM model.
        
        Returns
        -------
        clusters : list of lists
        clusters_names : list
        eigengene_matrix: DataFrame
        '''
        C = self._localMaximumQCM()
        clusters = self._merging_lmQCM(C)
        
        clusters_names = []
        for i in range(len(clusters)):
            mc = clusters[i]
            clusters_names.append(list(self.data_in.index.values[mc]))
        eigengene_matrix = np.zeros((len(clusters), self.data_in.shape[1]))
        for i in range(len(clusters_names)):
            gene = clusters_names[i]
            X = self.data_in.loc[gene, ]
            mu = np.nanmean(X, axis = 1) # rowMeans
            stddev = np.nanstd(X, axis = 1, ddof= 1) # ddof=1 provides unbiased estimation (1/(n-1))
            XNorm = (X.T-mu).T
            XNorm = (XNorm.T/stddev).T
            u, s, vh = np.linalg.svd(XNorm, full_matrices = False)
            eigenvector_first = vh[0,:]
            
            '''
            Compute the sign of the eigengene.
            
            1. Correlate the eigengene value with each of the gene's expression in that module across all samples used to generate the module.
            2. If >50% of the correlations is negative, then assign a – sign to the eigengene.
            3. If 50% or more correlation is positive, the eigengene remains positive.
            4. Output the eigene value table with the sign carried (if it is negative).
            '''
            mat_ = np.concatenate([eigenvector_first.reshape(1,-1), X.values])
            corr = np.corrcoef(mat_)[0,1:]
            negative_ratio = np.sum(corr < 0)/len(corr)
            if negative_ratio > 0.5:
                eigenvector_first = -eigenvector_first
            
            eigengene_matrix[i, ] = eigenvector_first
            
        eigengene_matrix = pd.DataFrame(eigengene_matrix, columns = self.data_in.columns)
        self.clusters = clusters
        self.clusters_names = clusters_names
        self.eigengene_matrix = eigengene_matrix
        return self.clusters, self.clusters_names, self.eigengene_matrix
    
    def predict(self,
                data_in: pd.DataFrame
                ) -> Tuple[pd.DataFrame, List[List]]:
        '''
        data_in : a P * N DataFrame with P index (genes) and N samples.
        
        Returns
        -------
        eigengene_matrix : DataFrame
        gene_not_existed : list of lists
        '''
        if not isinstance(data_in, pd.DataFrame):
            raise ValueError("Input data is not pandas DataFrame.")
        try:
            clusters_names = self.clusters_names
        except AttributeError:
            raise AttributeError("No fitted result found. Please try to fit a data.")
        else:
            eigengene_matrix = np.zeros((len(clusters_names), data_in.shape[1]))
            gene_not_existed = []
            for i in range(len(clusters_names)):
                gene = clusters_names[i]
                ne = [g for g in gene if g not in data_in.index]
                if len(ne) > 0:
                    warnings.warn('%d genes not existed in cluster %d.' % (len(ne), i) )
                gene_not_existed.append(ne)
                gene_overlapped = [g for g in gene if g in data_in.index]
                X = data_in.loc[gene_overlapped, ]
                mu = np.nanmean(X, axis = 1) # rowMeans
                stddev = np.nanstd(X, axis = 1, ddof= 1) # ddof=1 provides unbiased estimation (1/(n-1))
                XNorm = (X.T-mu).T
                XNorm = (XNorm.T/stddev).T
                u, s, vh = np.linalg.svd(XNorm, full_matrices = False)
                eigenvector_first = vh[0,:]
                
                '''
                Compute the sign of the eigengene.
                
                1. Correlate the eigengene value with each of the gene's expression in that module across all samples used to generate the module.
                2. If >50% of the correlations is negative, then assign a – sign to the eigengene.
                3. If 50% or more correlation is positive, the eigengene remains positive.
                4. Output the eigene value table with the sign carried (if it is negative).
                '''
                mat_ = np.concatenate([eigenvector_first.reshape(1,-1), X.values])
                corr = np.corrcoef(mat_)[0,1:]
                negative_ratio = np.sum(corr < 0)/len(corr)
                if negative_ratio > 0.5:
                    eigenvector_first = -eigenvector_first
                eigengene_matrix[i, ] = eigenvector_first
                
            eigengene_matrix = pd.DataFrame(eigengene_matrix, columns = data_in.columns)
            return eigengene_matrix, gene_not_existed
    
    
    

