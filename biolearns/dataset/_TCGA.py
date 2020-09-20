# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Tue Feb 11 12:40:52 2020
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

import pandas as pd
import numpy as np
import requests
import sys
import os
import re
import tarfile
import tempfile
'''
Parameters
----------
None
    
Returns
-------
None

Notes
-----
Data is downloaded directly from https://gdac.broadinstitute.org/.
The results here are in whole or part based upon data generated by 
the TCGA Research Network: https://www.cancer.gov/tcga.

References
----------
None
       
Examples
--------
>>> brca = TCGA('BRCA')
>>> brca.get_mRNAseq()
>>> brca.get_clinical()

'''

class TCGA():
    def __init__(self,
                 cohort = 'ACC',
                 download = True,
                 location = '/tmp'):
        self.cohort = cohort.upper()
        self.cohortdict = { 'ACC':'Adrenocortical carcinoma',
                            'BLCA':'Bladder urothelial carcinoma',
                            'BRCA':'Breast invasive carcinoma',
                            'CESC':'Cervical and endocervical cancers',
                            'CHOL':'Cholangiocarcinoma',
                            'COAD':'Colon adenocarcinoma',
                            'COADREAD':'Colorectal adenocarcinoma',
                            'DLBC':'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma',
                            'ESCA':'Esophageal carcinoma',
                            'FPPP':'FFPE Pilot Phase II',
                            'GBM':'Glioblastoma multiforme',
                            'GBMLGG':'Glioma',
                            'HNSC':'Head and Neck squamous cell carcinoma',
                            'KICH':'Kidney Chromophobe',
                            'KIPAN':'Pan-kidney cohort (KICH+KIRC+KIRP)',
                            'KIRC':'Kidney renal clear cell carcinoma',
                            'KIRP':'Kidney renal papillary cell carcinoma',
                            'LAML':'Acute Myeloid Leukemia',
                            'LGG':'Brain Lower Grade Glioma',
                            'LIHC':'Liver hepatocellular carcinoma',
                            'LUAD':'Lung adenocarcinoma',
                            'LUSC':'Lung squamous cell carcinoma',
                            'MESO':'Mesothelioma',
                            'OV':'Ovarian serous cystadenocarcinoma',
                            'PAAD':'Pancreatic adenocarcinoma',
                            'PCPG':'Pheochromocytoma and Paraganglioma',
                            'PRAD':'Prostate adenocarcinoma',
                            'READ':'Rectum adenocarcinoma',
                            'SARC':'Sarcoma',
                            'SKCM':'Skin Cutaneous Melanoma',
                            'STAD':'Stomach adenocarcinoma',
                            'STES':'Stomach and Esophageal carcinoma',
                            'TGCT':'Testicular Germ Cell Tumors',
                            'THCA':'Thyroid carcinoma',
                            'THYM':'Thymoma',
                            'UCEC':'Uterine Corpus Endometrial Carcinoma',
                            'UCS':'Uterine Carcinosarcoma',
                            'UVM':'Uveal Melanoma'}
        self.location = location
        if download:
#            self.tmpdir = tempfile.TemporaryDirectory()
            self.mRNAseq = self.get_mRNAseq()
            self.miRNAseq = self.get_miRNAseq()
            self.mRNA = None
            self.RPPA = None
            self.methylation = None
            self.clinical = self.get_clinical()
            self.overall_survival_time, self.overall_survival_event = self._get_overall_survival(death_censor = True)
        
    def get_mRNAseq(self):
        print('Retrieve mRNAseq from http://firebrowse.org/ ...')
        print('Cohort: %s (%s)' % (self.cohortdict[self.cohort], self.cohort))
        print('File type: illuminahiseq_rnaseqv2-RSEM_genes_normalized')

        link = 'https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/' + self.cohort + \
                '/20160128/gdac.broadinstitute.org_' + self.cohort + \
                '.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz'

        if self.cohort in ['FPPP']:
            print('No mRNAseq data.')
            return
            
        file_downloaded = os.path.join(self.location, self.cohort + '.mRNAseq.tar.gz')
        
        if not os.path.exists(file_downloaded):
            self._download_file(link, file_downloaded)
                            
        if self.cohort in ['COADREAD', 'ESCA', 'GBM', 'HNSC', 'KIPAN', 'KIRC', 'KIRP', 'LAML', \
                           'LIHC', 'OV', 'PCPG', 'READ', 'SKCM', 'THYM', 'UCEC', 'UCS']:
            skiprows = 1
        else:
            skiprows = None
        self.mRNAseq = pd.read_csv(file_downloaded, header=0, skiprows=skiprows, index_col=0, sep='\t', low_memory=False)
        self.mRNAseq = self.mRNAseq.loc[['|' in i for i in self.mRNAseq.index.values.astype(str)]] # keep only gene rows
        self.mRNAseq = self.mRNAseq.astype(float)
        self.mRNAseq.index = [re.split(r"\b\|\b", idx, 1)[0] for idx in self.mRNAseq.index.values.astype(str)]
        print('Done.')
#        os.remove(file_downloaded)
        return self.mRNAseq
    
    
    def get_miRNAseq(self):
        print('Retrieve miRNAseq from http://firebrowse.org/ ...')
        print('Cohort: %s (%s)' % (self.cohortdict[self.cohort], self.cohort))
        print('File type: illuminahiseq_mirnaseq-miR_gene_expression')
        
        link = 'https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/' + self.cohort + \
                '/20160128/gdac.broadinstitute.org_' + self.cohort + \
                '.Merge_mirnaseq__illuminahiseq_mirnaseq__bcgsc_ca__Level_3__miR_gene_expression__data.Level_3.2016012800.0.0.tar.gz'

        if self.cohort in ['LAML']:
            print('No miRNAseq data: illuminahiseq data not found. %s only has illuminaga data!' % self.cohort)
            return
            
        file_downloaded = os.path.join(self.location, self.cohort + '.miRNAseq.tar.gz')
        
        if not os.path.exists(file_downloaded):
            self._download_file(link, file_downloaded)
        
        if self.cohort in ['BLCA','CHOL','ESCA','HNSC','KIRP','LUAD','LUSC','OV','PAAD','READ','SKCM','STES','THCA','THYM','UVM']:
            skiprows = 1
        elif self.cohort in ['GBMLGG']:
            skiprows = 0
        else:
            skiprows = None
        self.miRNAseq = pd.read_csv(file_downloaded, header=None, skiprows=skiprows, index_col=0, sep='\t', low_memory=False)
        self.miRNAseq = self.miRNAseq.loc[:,self.miRNAseq.loc['miRNA_ID',:] == 'read_count']
        self.miRNAseq.columns = self.miRNAseq.iloc[0,:]
        self.miRNAseq = self.miRNAseq.iloc[2:] # drop first two rows
        self.miRNAseq = self.miRNAseq.loc[self.miRNAseq.index.notnull(),:] # drop NaN indexed rows
        self.miRNAseq = self.miRNAseq.astype(float)
        print('Done.')
#        os.remove(file_downloaded)
        return self.miRNAseq
        
    def get_clinical(self):
        print('Retrieve clinical from http://firebrowse.org/ ...')
        print('Cohort: %s (%s)' % (self.cohortdict[self.cohort], self.cohort))
        print('File type: Clinical_Pick_Tier1')

        link = 'https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/' + \
                self.cohort + '/20160128/gdac.broadinstitute.org_' + self.cohort + \
                '.Clinical_Pick_Tier1.Level_4.2016012800.0.0.tar.gz'

        file_downloaded = '/tmp/' + self.cohort + '.clinical.tar.gz'
        if not os.path.exists(file_downloaded):
            self._download_file(link, file_downloaded)
        with tarfile.open(file_downloaded) as f:
            clinical_file = f.extractfile('gdac.broadinstitute.org_'+self.cohort+'.Clinical_Pick_Tier1.Level_4.2016012800.0.0/' + self.cohort + '.clin.merged.picked.txt')
            self.clinical = pd.read_csv(clinical_file, header=0, index_col=0, sep='\t', low_memory=False).T
        self.clinical.index = [v.upper() for v in self.clinical.index.astype(str)]
        print('Patient barcode unique?', len(self.clinical.index) == len(np.unique(self.clinical.index)))
        print('Done.')
#        os.remove(file_downloaded)
        return self.clinical
    
    def _get_overall_survival(self, death_censor = True):
        if self.clinical is None:
            _ = self.get_clinical()
            
        colname = self.clinical.columns.values.astype(str)
        
        days_to_last_followup = self.clinical[colname[[i for i, d in enumerate(colname) if 'days_to_last_followup' in d]]].values.astype(str)
        days_to_death = self.clinical[colname[[i for i, d in enumerate(colname) if 'days_to_death' in d]]].values.astype(str)
        vital_status = self.clinical[colname[[i for i, d in enumerate(colname) if 'vital_status' in d]]].values.astype(str)
        
        print('Censorship: Dead = 1')
        e = vital_status.reshape(-1)
        e[[v in ['nan','NaN'] for v in vital_status]] = np.nan
        e = e.astype(int)
        if not death_censor:
            e = 1-e
        
        t = days_to_last_followup
        t[np.where(e == 1)] = days_to_death[np.where(e == 1)]
        t[[v in ['nan','NaN'] for v in t]] = np.nan
        t = t.reshape(-1).astype(float)
        
        return t, e
            
    def _download_file(self, link, filename):
        if not os.path.exists(filename):
            with open(filename, "wb") as f:
#                    print("Downloading %s to %s" % (self.cohort, filename))
                    response = requests.get(link, stream=True)
                    total_length = response.headers.get('content-length')
                    if total_length is None: # no content length header
                        f.write(response.content)
                    else:
                        dl = 0
                        total_length = int(total_length)
                        for data in response.iter_content(chunk_size=4096):
                            dl += len(data)
                            f.write(data)
                            done = int(50 * dl / total_length)
                            sys.stdout.write("\r[%s%s]" % ('=' * done + '>', ' ' * (50-done)) )    
                            sys.stdout.flush()
