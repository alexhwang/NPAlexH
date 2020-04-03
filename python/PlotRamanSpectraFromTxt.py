# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:49:49 2019

@author: aykh2
"""

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 16
rcParams['legend.fontsize'] = 12
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
import pandas as pd
import os
import glob
import copy
import h5py

def basic_SERS_plot(filename, title, figurenum, xlim, ylim, power, int_time, savefilename = 'tmp.png', accum = 1, leg = [], ylabel = 'SERS intensity (counts mW$^{-1}$ s$^{-1}$)', usecols = [0,1]):
    fileobj = pd.read_csv(filename, sep='\t', usecols=usecols, names=['Wavenumber', 'Intensity'], skiprows = 1)
    fileobj_np = fileobj.to_numpy()
    wn = fileobj_np[:, 0]
    int = fileobj_np[:, 1]
    int_mW_s = int / power / int_time / accum
    
    temp = int
    plt.figure(figurenum)
    plt.plot(wn, int_mW_s)
    plt.xlabel('Wavenumber (cm$^{-1}`$)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.legend(leg)
    plt.savefig(savefilename)

def ReadTxtFileMultiple(filename, usecols = [0, 1, 2]):
    fileobj = pd.read_csv(filename, sep='\t', usecols=usecols, names=['Time', 'Wavenumber', 'Intensity'], skiprows = 1)
    fileobj_pivoted = fileobj.pivot(index='Wavenumber', columns='Time')
    int_matrix = fileobj_pivoted.to_numpy()
    wn = fileobj_pivoted.index.to_numpy()
    
    return (wn, int_matrix)

def ReadTxtFileSingle(filename, usecols = [0, 1]):
    fileobj = pd.read_csv(filename, sep='\t', usecols=usecols, names=['Wavenumber', 'Intensity'], skiprows = 1)
    fileobj_np = fileobj.to_numpy()
    wn = fileobj_np[:, 0]
    int = fileobj_np[:, 1]
    
    return (wn, int)

def CombineIntMatrixCols(int_matrix, n, method='sum'):
    int_matrix_shape = np.shape(int_matrix)
    acq = (int_matrix_shape[1])
    int_matrix_compressed = np.zeros([(int_matrix_shape[0]), acq / n])
    
    for i in range(0, acq / n):
        if method=='sum':
            int_matrix_compressed[:, i] = np.array(np.sum(int_matrix[:, i*n:i*n+n], 1))
        elif method=='avg':
            int_matrix_compressed[:, i] = np.array(np.mean(int_matrix[:, i*n:i*n+n], 1))
    return int_matrix_compressed
    
def SERS_Spectra_PlotMultAcqSingleFile(filename, title, figurenum, xlim, ylim, power = 1, int_time = 1, 
                                       savefilename = 'tmp.png',
                              leg = [], ylabel = 'SERS intensity (counts mW$^{-1}$ s$^{-1}$)', to_plot = [0], usecols = [0, 1, 2], combine = False, method='avg'):
    
    (wn, int_matrix) = ReadTxtFileMultiple(filename)
    
    if combine != False:
        int_matrix = CombineIntMatrixCols(int_matrix, n = combine, method=method)
    
    PlotMultipleHelper(wn, int_matrix, title, figurenum, xlim, ylim, power, int_time, savefilename, leg, ylabel, to_plot)

def SERS_Spectra_PlotMultAcqDiffFiles(filename_lst, title, figurenum, xlim, ylim, power=1, int_time=1, savefilename='tmp.png', 
                                      leg = [], ylabel = 'SERS intensity (counts mW$^{-1}$ s$^{-1}$)', to_plot = [0], combine = False, 
                                      offset=0, initial_offset=0, medfilt=1, NColLeg=2, wn_size=2549, plotoptions=None, yscale='linear', method='avg'):
    if type(wn_size) == int:
        
        num_acq = len(filename_lst)
        int_matrix = np.zeros([wn_size, num_acq])
        wn = np.zeros([wn_size, 1])
        for i in range(0, num_acq):
            filename = filename_lst[i]
            (wn, intensity) = ReadTxtFileSingle(filename_lst[i])
            int_matrix[:, i] = intensity
            
        if combine != False:
            int_matrix = CombineIntMatrixCols(int_matrix, n = combine, method='avg')
        
        PlotMultipleHelper(wn, int_matrix, title, figurenum, xlim, ylim, power, int_time, savefilename, leg, ylabel, to_plot,
                           offset, initial_offset, medfilt, NColLeg, plotoptions, yscale)
        
    elif type(wn_size) == list:
        num_acq = len(filename_lst)
        
        for i in range(0, num_acq):
            int_matrix = np.zeros([wn_size[i], 1])
            wn = np.zeros([wn_size[i], 1])
            filename = filename_lst[i]
            (wn, intensity) = ReadTxtFileSingle(filename_lst[i])
            int_matrix[:, 0] = intensity
        
            PlotMultipleHelper(wn, int_matrix, title, figurenum, xlim, ylim, power, int_time, savefilename, leg, ylabel, to_plot, 
                               offset, initial_offset, medfilt, NColLeg, [plotoptions[i]], yscale)
        
    
def PlotMultipleHelper(wn, int_matrix, title, figurenum, xlim, ylim, power = 1, int_time = 1, savefilename = 'tmp.png',
                              leg = [], ylabel = 'SERS intensity (counts mW$^{-1}$ s$^{-1}$)', to_plot = [0], offset=0, initial_offset=0, medfilt=1,
                              NColLeg=2, plotoptions=None, yscale='linear'):
    if plotoptions==None:
        plotoptions = ['']*len(to_plot)
        
    plt.figure(figurenum)
    for i in to_plot:
        if type(power) == list:
            data_to_plot = np.add(int_matrix[:, i] / power[i] / int_time[i], i*offset+initial_offset);
        else:
            data_to_plot = np.add(int_matrix[:, i] / power / int_time, i*offset+initial_offset);
        
        data_to_plot = signal.medfilt(data_to_plot, medfilt)
        plt.plot(wn, data_to_plot, plotoptions[i])
        plt.xlabel('Wavenumber (cm$^{-1}`$)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.axes().minorticks_on()
        if yscale=='log':
            plt.yscale('log')

    plt.legend(leg, ncol = NColLeg)
    if (savefilename != 'tmp.png') & (savefilename != ''):
        plt.savefig(savefilename)
    
def PlotAllTxtInFolder(folder, startfigurenum=1):
    filenames = glob.glob(folder+'\\*.txt')
    figurenum = startfigurenum
    for filename in filenames:
        basic_SERS_plot(filename, filename[len(folder)+1:], figurenum, xlim=None, ylim=None, power=1, int_time=1, 
                        savefilename = 'tmp.png', ylabel = 'SERS intensity (counts)')
        figurenum += 1

def FilenamesToData(filename_lst, combine=False, delimiter='\t'):
    
    if combine == False:
        N = len(filename_lst)
        spectral_data = []
        
        for i in range(0,N):
            fileobj = pd.read_csv(filename_lst[i], sep=delimiter, usecols=[0,1], names=['Wavenumber', 'Intensity'], skiprows = 1)
            fileobj_np = fileobj.to_numpy()
            wn = np.flipud(fileobj_np[:, 0])
            int = np.flipud(fileobj_np[:, 1])
            spectral_data.append((wn, int))
            
        return spectral_data
    elif combine:
        N = len(filename_lst)
        wn=[]
        intensity=[]
        
        for i in range(0,N):
            fileobj = pd.read_csv(filename_lst[i], sep='\t', usecols=[0,1], names=['Wavenumber', 'Intensity'], skiprows = 1)
            fileobj_np = fileobj.to_numpy()
            wn = np.concatenate([wn, np.flipud(fileobj_np[:, 0])]) if i!=0 else np.flipud(fileobj_np[:, 0])
            intensity = np.concatenate([intensity, np.flipud(fileobj_np[:, 1])]) if i!=0 else np.flipud(fileobj_np[:, 1])
            
        return (np.sort(wn),intensity[np.argsort(wn)])    
    

def PlotData(data, norm=1.0, offset=0, *argv, **kwargs):    
    N = len(data)
    for i in range(N):
        current = data[i]
        current_norm = norm
        if type(norm) == list:
            current_norm = norm[i]
        current_raman = np.divide(current[1], current_norm)
        current_raman += offset * i
        plt.plot(current[0], current_raman, *argv, **kwargs)
        plt.axes().minorticks_on()

def QuarticPoly(x, QuarticCoeffs):
    return QuarticCoeffs[0] + QuarticCoeffs[1]*x + QuarticCoeffs[2]*(x**2) + QuarticCoeffs[3]*(x**3) + QuarticCoeffs[4]*(x**4)

def PlotDataBGSub(data, QuarticBG, norm=1.0, offset=0, *argv, **kwargs):
    BG = QuarticPoly(data[0][0], QuarticBG)
    data_BGsub = data
    data_BGsub[0][1] = data[0][1] - BG
    PlotData(data_BGsub, norm, offset, *argv, **kwargs)

def PlotDataAvg(data, norm=1.0, offset=0, *argv, **kwargs):
    N = len(data)
    wn = data[0][0]
    
    avged_int = data[0][1]
    
    for i in range(1,N):
        avged_int = np.add(avged_int, data[i][1])
        
    avged_int /= N
        
    PlotData([(wn, avged_int)], norm, offset, *argv, **kwargs)
    
    return((wn, avged_int))
    
def DataAvg(data):
    N = len(data)
    wn = data[0][0]
    
    avged_int = data[0][1]
    
    for i in range(1,N):
        avged_int = np.add(avged_int, data[i][1])
        
    avged_int /= N
            
    return[(wn, avged_int)]
    
def medfilt(data, n=9, wn_include='all', padding=0):
    smooth_results = []
    for i in range(0, len(data)):
        wn = data[i][0]
        raman = data[i][1]
        
        if wn_include != 'all':  # only filter specified parts
            wn_cut_bot_idx = np.where(np.absolute(wn-wn_include[0]) <= 1)[0][0]
            wn_cut_top_idx = np.where(np.absolute(wn-wn_include[1]) <= 1)[0][0]
            
            raman_cut = raman[(wn_cut_bot_idx):(wn_cut_top_idx)]
            
            raman_cut_medfilt = scipy.signal.medfilt(raman_cut, n)
            raman_medfilt = copy.copy(raman)
            raman_medfilt[(wn_cut_bot_idx+padding):(wn_cut_top_idx-padding)] = raman_cut_medfilt[padding:-padding]
        else:
            raman_medfilt = scipy.signal.medfilt(raman, n)
            
        smooth_results.append((wn, raman_medfilt))
    return smooth_results

def savgol_filter(data, window_length=3, polyorder=1):
    savgol_results=[]
    for i in range(0, len(data)):
        wn=data[i][0]
        raman = data[i][1]
        
        raman_savgol = scipy.signal.savgol_filter(raman, window_length, polyorder)
        
        savgol_results.append((wn, raman_savgol))
    return savgol_results

def DataToTxt(data, filename):
    array_version = np.hstack([data[0].reshape(-1,1), data[1].reshape(-1,1)])
    np.savetxt(filename, array_version)

def NormToPeak(data, peak_wn):
    norm_results = []
    
    wn_0 = data[0][0]
    peak_wn_idx = np.where(np.absolute(wn_0-peak_wn) <= 1)[0][0]
    
    for i in range(len(data)):
        wn = data[i][0]
        raman = data[i][1]
        raman_norm = raman / raman[peak_wn_idx]
        
        norm_results.append((wn, raman_norm))
    return norm_results
        
def data_statistics(data):
    data_0 = data[0]
    wn = data_0[0]
    L = len(wn)
    N = len(data)
    
    data_array = np.zeros([L, N])
    
    for i in range(N):
        data_array[:, i] = data[i][1]
    
    std = np.std(data_array, 1)
    means = np.mean(data_array, 1)
    medians = np.percentile(data_array, 50, 1)
    percentile90 = np.percentile(data_array, 90, 1)
    percentile10 = np.percentile(data_array, 10, 1)
    
    return (wn, means, std, medians, percentile10, percentile90)