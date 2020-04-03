# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 06:51:08 2019

@author: aykh2
"""

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
    
#def PlotAllTxtInFolder(folder, startfigurenum=1):
#    filenames = glob.glob(folder+'\\*.txt')
#    figurenum = startfigurenum
#    for filename in filenames:
#        basic_SERS_plot(filename, filename[len(folder)+1:], figurenum, xlim=None, ylim=None, power=1, int_time=1, 
#                        savefilename = 'tmp.png', ylabel = 'SERS intensity (counts)')
#        figurenum += 1

def FilenamesToData(filename_lst, combine=False, delimiter='\t', flip=True, separated=False):
    
    if separated==True:
        N = len(filename_lst)
        
        spectral_data_list = []
        for i in range(N):
            fileobj = pd.read_csv(filename_lst[i], sep=delimiter, usecols=[0,1], names=['Wavenumber', 'Intensity'], skiprows = 1)
            fileobj_np = fileobj.to_numpy()
            
            if flip:
                wn = np.flipud(fileobj_np[:, 0])
                intensity = np.flipud(fileobj_np[:, 1])
            else:
                wn = fileobj_np[:, 0]
                intensity = fileobj_np[:, 1]
        
            spectral_data_list.append((wn, intensity.squeeze()))
        return spectral_data_list
    
    if combine == False:
        N = len(filename_lst)
        
        first = pd.read_csv(filename_lst[0], sep=delimiter, usecols=[0,1], names=['Wavenumber', 'Intensity'], skiprows = 1)
        first_np = first.to_numpy()
        num_datapts = np.shape(first_np)[0]
        num_datanames = len(filename_lst)
        
        spectral_data = np.zeros([num_datapts, num_datanames])
        wn = np.zeros(num_datapts)
        
        for i in range(0,N):
            fileobj = pd.read_csv(filename_lst[i], sep=delimiter, usecols=[0,1], names=['Wavenumber', 'Intensity'], skiprows = 1)
            fileobj_np = fileobj.to_numpy()
            
            if flip:
                wn = np.flipud(fileobj_np[:, 0])
                spectral_data[:, i] = np.flipud(fileobj_np[:, 1])
            else:
                wn = fileobj_np[:, 0]
                spectral_data[:, i] = fileobj_np[:, 1]
            
        return (wn, spectral_data.squeeze())
    
    
    
def FilenamesToDataMultiple(filename_lst, combine=False, delimiter='\t', flip=False):
    
    N = len(filename_lst)
    
    fileobj = pd.read_csv(filename_lst[0], sep='\t', usecols=[0,1,2], names=['Time', 'Wavenumber', 'Intensity'], skiprows = 1)
    fileobj_pivoted = fileobj.pivot(index='Wavenumber', columns='Time')
        
    if flip:
        int_matrix = np.flip(fileobj_pivoted.to_numpy(), 0)
        wn = np.flip(fileobj_pivoted.index.to_numpy(), 0)
    else:
        int_matrix = fileobj_pivoted.to_numpy()
        wn = fileobj_pivoted.index.to_numpy()
    
    for i in np.arange(1,N):
        
        fileobj = pd.read_csv(filename_lst[i], sep='\t', usecols=[0,1,2], names=['Time', 'Wavenumber', 'Intensity'], skiprows = 1)
        fileobj_pivoted = fileobj.pivot(index='Wavenumber', columns='Time')
        
        if flip:
            int_matrix = np.concatenate((int_matrix, np.flip(fileobj_pivoted.to_numpy())), 1)
            wn = np.flip(fileobj_pivoted.index.to_numpy(), 0)
        else:
            int_matrix = np.concatenate((int_matrix, (fileobj_pivoted.to_numpy())), 1)
            wn = fileobj_pivoted.index.to_numpy()
    
    return (wn, int_matrix)
    
def FilenameToDataH5(filename, inst='OceanOpticsSpectrometer', datanames='', combine=False):
    """
    Input: h5 filename
    Output: (wn, int_matrix)
    
    int_matrix is len(wn) x num_files
    """    
    
    f = h5py.File(filename, 'r')
    inst_group = f[inst]
    
    first = inst_group[list(inst_group.keys())[0]]
    wavelength = first.attrs['wavelengths']
    
    num_datapts = first.size
    num_datanames = len(datanames)
    
    data_matrix = np.zeros([num_datapts, num_datanames])
    
    for i in range(num_datanames):
        current = inst_group[datanames[i]]
        data_matrix[:, i] = current[0:current.size]
        
    return (wavelength, data_matrix.squeeze())
        
def PlotData(xaxis, data_matrix, norm=1.0, offset=0, *argv, **kwargs):    
    
    dim_data_matrix = len(np.shape(data_matrix))
    if dim_data_matrix==2:
        N = np.shape(data_matrix)[1]
    else:
        N = 1
        
    for i in range(N):
        
        if dim_data_matrix == 2:
            current = data_matrix[:, i]
        else:
            current = data_matrix
            
        current_norm = norm
        if type(norm) == list:
            current_norm = norm[i]
        current_raman = np.divide(current, current_norm)
        current_raman += offset * i
        plt.plot(xaxis, current_raman, *argv, **kwargs)
        plt.axes().minorticks_on()

def AvgData(data_matrix):
    
    return np.mean(data_matrix, 1)

def medfilt(data_matrix, N=1):
    
    data_matrix_shape = np.shape(data_matrix)
    Ncol = data_matrix_shape[1]
    data_matrix_filtered = np.zeros(data_matrix_shape)
    
    for i in range(Ncol):
        current = data_matrix[:, i]
        data_matrix_filtered[:, i] = scipy.signal.medfilt(current, N)
        
    return data_matrix_filtered

def modified_z_score(intensity):
 median_int = np.median(intensity)
 mad_int = np.median([np.abs(intensity - median_int)])
 modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
 return abs(modified_z_scores)


def RemoveCRAcq(data_matrix, threshold):
    to_keep = []
    for j in np.arange(np.shape(data_matrix)[1]):
        current = data_matrix[:, j]
        max_z_score = np.max(modified_z_score(current))
        if max_z_score < threshold:
            to_keep.append(j)
    print('Removing:' + str(np.setdiff1d(np.arange(np.shape(data_matrix)[1]), to_keep)))  
              
    return data_matrix[:, to_keep]

def FilterCR(data_matrix, threshold, order):
    filtered_data_matrix = np.zeros(np.shape(data_matrix))
    for j in np.arange(np.shape(data_matrix)[1]):
        current = data_matrix[:, j]
        current_z_score = modified_z_score(current)
        spikes = current_z_score > threshold
        max_z_score = np.max(current_z_score)
        if max_z_score > threshold:
            current = FilterCRHelper(current, spikes, order)
        filtered_data_matrix[:, j] = current
            
    return filtered_data_matrix

def FilterCRHelper(intensity, spikes, order):
    int_out = intensity.copy()
    
    for i in np.arange(len(spikes)):
        if spikes[i] != 0:
            w = np.arange(i-order, i+1+order)
            w2 = w[spikes[w] == 0]
            int_out[i] = np.mean(intensity[w2])
    
    return int_out

def AddOffsets(data_matrix, offset):
    data_matrix_shape = np.shape(data_matrix)
    offsets = np.tile(np.arange(0, data_matrix_shape[1])*offset, (data_matrix_shape[0], 1))
    
    return data_matrix+offsets

def mpl_defaults():
    plt.xlim([0, 4000])
    plt.ylim([0, 10000])
    plt.xlabel('Wavenumber (cm-1)')
    plt.ylabel('Counts')
    plt.minorticks_on()

def calculate_max_after_100wn(x, y):
    if y.ndim == 1:
        return np.amax(y[np.where(x>100)])
    elif y.ndim == 2:
        return np.amax(y[np.where(x>100), :])

def from_h5_plot_all(h5data):
    f_loaded = h5data['All Raw']
    spectra_names_lst = list(f_loaded.keys())
    num_spectra = len(spectra_names_lst)
    for i in range(num_spectra):
        current_spectra_name = spectra_names_lst[i]
        current_data = f_loaded[current_spectra_name]
        current_data_raman = np.transpose(np.array(current_data))
        current_data_x = current_data.attrs['x']
        current_data_attrs = current_data.attrs
        
#         print(current_data_attrs.keys())
        
        plot_title = (current_spectra_name + '\n' + 
                      'Laser:' + str(current_data_attrs['Laser Wavelength']) +
                      '; Power:' + str(int(current_data_attrs['Laser Power (%%)'])) + '%' + 
                      ' (' + str(current_data_attrs['Laser Power (mW)']) + 'mW)' +
                      '; Time:' + str(current_data_attrs['Integration Time']/1e3) + #
                      '; Grating:' + str(current_data_attrs['Grating']) + '\n' +
                      current_data_attrs['creation_time'])
        
        plt.figure()
        plt.plot(current_data_x, current_data_raman)
        plt.title(plot_title, fontsize='x-small')
        
        ylim_max = calculate_max_after_100wn(current_data_x, current_data_raman) * 1.1
        plt.xlim([0, 4000])
        plt.ylim([0, ylim_max])
        plt.xlabel('Raman shift (cm-1)')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.show()    
