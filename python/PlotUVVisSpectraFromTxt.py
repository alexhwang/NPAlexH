# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:49:49 2019

@author: aykh2
"""

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 16
rcParams['legend.fontsize'] = 14
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def basic_UVVis_plot(filename, title, figurenum, xlim, ylim, savefilename, leg = [], ylabel = 'Absorbance'):
    fileobj = pd.read_csv(filename, sep='\t', usecols=[0,1], skiprows = 1)
    fileobj_np = fileobj.to_numpy()
    wn = fileobj_np[:, 0]
    int = fileobj_np[:, 1]
    
    plt.figure(figurenum)
    plt.plot(wn, int)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(leg)
    plt.savefig(savefilename)

plt.close('all')

### Au Only

basic_UVVis_plot('export-data.txt', '2019-10-22 David AuNP-CB[5] Aggregates', 1, None, None,
                 'DavidNPCB5Aggregate-UVVis.png')
