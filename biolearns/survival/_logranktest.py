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

def logranktest(hazard_ratio, time, event):
    