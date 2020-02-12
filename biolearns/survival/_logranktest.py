# Copyright 2020 Zhi Huang.  All rights reserved
# Created on Tue Feb 12 16:00:51 2020
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
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter

def logranktest(hazard_ratio, time, event):
    groups = (hazard_ratio > np.median(hazard_ratio)).astype(int)
    # 0: low hazard ratio
    # 1: high hazard ratio
    
    logrank_results = logrank_test(time[groups == 0], time[groups == 1], event_observed_A=event[groups == 0], event_observed_B=event[groups == 1])
    p_value = logrank_results.p_value
#    test_statistic = logrank_results.test_statistic
    # plots
    kmf = KaplanMeierFitter()
    fig = plt.figure()
    ax = plt.subplot(111)
    kmf.fit(time[groups == 0], event_observed=event[groups == 0], label='Group 1')
    kmf.plot(ax=ax, show_censors = True, ci_show = False, color = 'black')
    kmf.fit(time[groups == 1], event_observed=event[groups == 1], label='Group 2')
    kmf.plot(ax=ax, show_censors = True, ci_show = False, color = 'red')
    ax.text(0, ax.get_ylim()[0]+0.05, s = '  p-value = %.4e' % p_value, fontsize = 14)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival probability')
    ax.set_title('Survival Plot', fontsize = 14)
    
    return logrank_results, fig
