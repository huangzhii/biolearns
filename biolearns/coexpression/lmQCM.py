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
>>> data_in = pd.read_csv('~/Desktop/data.csv', index_col = 0) # index: Features (Genes); columns: samples.
>>> lobject = lmQCM(data_in, gamma = 0.55, t = 1, lambdaa = 1, beta = 0.4, 
              minClusterSize = 2, CCmethod = "pearson", normalization = True)
>>> lobject.fit()
>>> lobject.clusters
>>> lobject.clusters_names
>>> lobject.eigengene_matrix
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr


class lmQCM():
    def __init__(self, data_in = None, gamma = 0.55, t = 1, lambdaa = 1, beta = 0.4, 
                     minClusterSize = 10, CCmethod = "pearson", normalization = False):
        self.data_in = data_in
        self.gamma = gamma
        self.t = t
        self.lambdaa = lambdaa
        self.beta = beta
        self.minClusterSize = minClusterSize
        self.CCmethod = CCmethod
        self.normalization = normalization
    
    def localMaximumQCM(self, cMatrix):
        C = []
        nRow = cMatrix.shape[0]
        maxV = np.max(cMatrix, axis = 0)
        maxInd = np.argmax(cMatrix, axis = 1)
        lm_ind = np.where(maxV == np.max(cMatrix[maxInd,], axis = 1))[0]
        maxEdges = np.stack((maxInd[lm_ind], lm_ind)).T
        maxW = maxV[lm_ind]
        sortMaxV = np.sort(maxW, kind='mergesort')[::-1] # decreasing
        sortMaxInd = np.argsort(maxW, kind='mergesort')[::-1]
        sortMaxEdges = maxEdges[sortMaxInd, ]
        print("Number of Maximum Edges: %d" % len(sortMaxInd))
        currentInit = 1
        noNewInit = 0
        
        nodesInCluster = []
        while currentInit <= len(sortMaxInd) and noNewInit == 0:
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
                        neighborWeights = np.sum(cMatrix[newCluster,:][:,remainInd], axis = 0)
                        maxNeighborWeight = max(neighborWeights)
                        maxNeighborInd = np.argmax(neighborWeights)
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
        return(C)
        
    def merging_lmQCM(self, C):
        sizeC = [len(i) for i in C]
        sortInd = np.argsort(sizeC, kind='mergesort')[::-1]
        mergedCluster = [C[i] for i in sortInd if len(C[i]) >= self.minClusterSize]
        mergeOccur = 1
        currentInd = -1
        print(" %d Modules before merging." % len(mergedCluster))
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
            sortMergedInd = np.argsort(sizeMergedCluster, kind='mergesort')[::-1]
            mergedCluster = [mergedCluster[i] for i in sortMergedInd]
            currentInd = 0
        print(" %d Modules remain after merging." % len(mergedCluster))
        return mergedCluster
        
    def fit(self):
        print("Calculating massive correlation coefficient ...")
        if self.CCmethod.lower() == "pearson": cMatrix = np.corrcoef(self.data_in)
        if self.CCmethod.lower() == "spearman": cMatrix = spearmanr(self.data_in.T).correlation
        np.fill_diagonal(cMatrix, 0)
        if self.normalization:
            cMatrix = np.abs(cMatrix)
            D = np.sum(cMatrix, axis = 0)
            D_half = 1.0/np.sqrt(D)
            cMatrix = np.multiply(np.multiply(cMatrix, D_half).T, D_half)
        C = self.localMaximumQCM(cMatrix)
        clusters = self.merging_lmQCM(C)
        clusters_names = []
        for i in range(len(clusters)):
            mc = clusters[i]
            clusters_names.append(list(self.data_in.index.values[mc]))
        eigengene_matrix = np.zeros((len(clusters), self.data_in.shape[1]))
        for i in range(len(clusters_names)):
            geneID = clusters_names[i]
            X = self.data_in.loc[geneID, ]
            mu = np.nanmean(X, axis = 1) # rowMeans
            stddev = np.nanstd(X, axis = 1, ddof= 1) # ddof=1 provides unbiased estimation (1/(n-1))
            XNorm = (X.T-mu).T
            XNorm = (XNorm.T/stddev).T
            u, s, vh = np.linalg.svd(XNorm, full_matrices = False)
            eigengene_matrix[i, ] = vh[0,:]
        eigengene_matrix = pd.DataFrame(eigengene_matrix, columns = self.data_in.columns)
        self.clusters = clusters
        self.clusters_names = clusters_names
        self.eigengene_matrix = eigengene_matrix

