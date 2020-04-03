# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:00:53 2020

@author: aykh2
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
import pandas as pd
import os
import glob
import copy
import h5py
import copy
import nplab.analysis.NPoM_Dark_Field_categorization.DF_Multipeakfit as DF_Multipeakfit
import matplotlib as mpl
import SPC_to_h5_ayh
import AlexHRaman as AHR

#########################################
### RamanSpectrum class

class RamanSpectrum:
    def __init__(self, wn, spec_data, attrs={}):
        if wn[0] < wn[1]:
            self.wn = np.squeeze(wn)
            spec_data_tmp = np.reshape(spec_data, (-1, np.size(wn)))
            self.spec_data = spec_data_tmp[~np.all(spec_data_tmp==0, axis=1)]
        else:
            self.wn = np.flip(np.squeeze(wn))
            spec_data_tmp = np.fliplr(np.reshape(spec_data, (-1, np.size(wn))))
            self.spec_data = spec_data_tmp[~np.all(spec_data_tmp==0, axis=1)]
        self.attrs=attrs

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.wn, np.transpose(self.spec_data))
        plt.minorticks_on()
        plt.xlim([0, 4000])
        plt.ylim([0, calculate_max_after_100wn(self.wn, self.spec_data)])
        return (fig, ax)
    
    def plot_avg(self, avg_slice=np.s_[:,:]):
        avged = np.mean(np.transpose(self.spec_data[avg_slice]), 1)
        
        fig, ax = plt.subplots()
        ax.plot(self.wn, avged)
        plt.minorticks_on()
        plt.xlim([0, 4000])
        plt.ylim([0, calculate_max_after_100wn(self.wn, self.spec_data)])
        return (fig, ax)
    
    def get_avg(self, avg_slice=np.s_[:,:]):
        sliced = self.spec_data[avg_slice]
        if np.ndim(sliced) == 1:
            return np.reshape(sliced, (1, -1))
        avged = np.mean((sliced), 0)
        return np.reshape(avged, (1, -1))
    
    def replace_avged(self, avg_slice=np.s_[:,:]):
        
        avged = self.get_avg(avg_slice)
        self_copy = copy.deepcopy(self)
        self_copy.spec_data = avged
        return self_copy
    
    def add_offsets(self, offset=0):
        
        self_copy = copy.deepcopy(self)
        for i in range(self_copy.spec_data.shape[0]):
            self_copy.spec_data[i, :] = self_copy.spec_data[i, :] + offset*i
        return self_copy
    
    def removeCRs(self, threshold=5, window_size=5):
        self_copy = copy.deepcopy(self)
        for i in range(self_copy.spec_data.shape[0]):
            self_copy.spec_data[i,:] = removeCRArray(self_copy.wn, self_copy.spec_data[i,:], threshold=threshold, window_size=window_size, start_wn=100)
        return self_copy

    def removeCRs_charlie(self, reference=1, factor=15):
        self_copy = copy.deepcopy(self)
        for i in range(self.spec_data.shape[0]):
            self_copy.spec_data[i, :] = DF_Multipeakfit.removeCosmicRays(self.wn, self_copy.spec_data[i,:], reference=reference, factor=factor)
        return self_copy

    def median_contract_by_3(self):
        self_copy = copy.deepcopy(self)
        spec_data_shape = np.shape(self_copy.spec_data)
        num_wn = spec_data_shape[1]
        num_spectra = spec_data_shape[0]

        j_max = np.floor(num_spectra/3)-1
        self_copy.spec_data = np.zeros((int(j_max)+1, num_wn))

        for j in np.arange(int(j_max+1)):
            self_copy.spec_data[j, :] = np.median(self.spec_data[(3*j):(3*(j+1)), :], 0)

        return self_copy

    def medfilt_by_acquisition(self):
        if (np.shape(self.spec_data)[0] == 1):
            return self
        return RamanSpectrum(self.wn, scipy.signal.medfilt(self.spec_data, [3, 1]))

    def export_to_txt(self, filename):
        np.savetxt(filename, np.vstack((np.reshape(self.wn, (1, -1)), self.spec_data)).T, delimiter=',')
        print('Exported to '+filename)

    def reduce_wn_region(self, new_wn_region):
        new_wn_idx_1 = min(enumerate(self.wn), key=lambda x:abs(x[1]-new_wn_region[0]))
        new_wn_idx_2 = min(enumerate(self.wn), key=lambda x:abs(x[1]-new_wn_region[1]))
        
        new_wn_idx_low = min(new_wn_idx_1[0], new_wn_idx_2[0])
        new_wn_idx_high = max(new_wn_idx_1[0], new_wn_idx_2[0])
        
        return RamanSpectrum(self.wn[new_wn_idx_low: new_wn_idx_high+1],
            self.spec_data[:, new_wn_idx_low: new_wn_idx_high+1])

    def cut_wn_region(self, cut_wn_region):
        cut_wn_idx_1 = min(enumerate(self.wn), key=lambda x:abs(x[1]-cut_wn_region[0]))
        cut_wn_idx_2 = min(enumerate(self.wn), key=lambda x:abs(x[1]-cut_wn_region[1]))
        
        cut_wn_idx_low = min(cut_wn_idx_1[0], cut_wn_idx_2[0])
        cut_wn_idx_high = max(cut_wn_idx_1[0], cut_wn_idx_2[0])
        
        all_indices = np.arange(np.size(self.wn))
        cut_indices = np.arange(cut_wn_idx_low, cut_wn_idx_high+1)
        all_but_cut_indices = np.setdiff1d(all_indices, cut_indices)

        return RamanSpectrum(self.wn[all_but_cut_indices],
            self.spec_data[:, all_but_cut_indices])

    def flip_wn(self):
        return RamanSpectrum(np.flip(self.wn), np.fliplr(self.spec_data))

    def new_rows(self, new_rows):
        return RamanSpectrum(self.wn, self.spec_data[new_rows, :])

#########################################


#########################################
### Functions to read h5 files

def from_h5file_make_dict(fileloc):
    with h5py.File(fileloc, 'r') as f:
        if 'All Raw' in f:
            output_dict = from_h5obj_make_dict(f['All Raw'], shorten_spec_names=True)
        else:
            output_dict = from_h5obj_make_dict(f)
        
    return output_dict

def from_h5obj_make_dict(h5obj, shorten_spec_names=False, create_times_lst=False):
    output_dict = {}
    i=0
    times_lst = []
    for key,value in h5obj.items():
        if type(value) == h5py.Dataset:
            keystring = 'spec_'+str(i) if shorten_spec_names else str(key)
            output_dict[keystring] = RamanSpectrum(value.attrs['x'], 
                np.array(value),
                dict(value.attrs))
            times_lst.append(value.attrs['creation_time'])
            i=i+1
        elif type(value) == h5py.Group:
            output_dict[key] = from_h5obj_make_dict(value)
    if create_times_lst:
        return output_dict, times_lst
    else:
        return output_dict
    

def process_raman_data(directory, spc_to_h5=True, plot_all=True, create_times_lst=False):
    os.chdir(directory)
    if spc_to_h5:
        SPC_to_h5_ayh.run(os.getcwd())
    with open(os.path.basename(os.getcwd())+'-raman-data.h5', 'rb') as f:
        current_data = h5py.File(f, 'r')
        # current_data_dict = AHR.from_h5_create_dict(current_data, create_times_lst)
        current_data_result = AHR.from_h5obj_make_dict(current_data['All Raw'], shorten_spec_names=True, create_times_lst=create_times_lst)
        if plot_all:
            AHR.from_h5_plot_all(current_data)
    return current_data_result

def from_h5_plot_all(h5data):
    f_loaded = h5data['All Raw']
    spectra_names_lst = list(f_loaded.keys())
    num_spectra = len(spectra_names_lst)
    times_lst = []
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

        times_lst.append(current_data_attrs['creation_time'])
        
        plt.figure()
        plt.plot(current_data_x, current_data_raman)
        plt.title(plot_title, fontsize='x-small')
        
        ylim_max = calculate_max_after_100wn(current_data_x, np.transpose(current_data_raman)) * 1.1
        plt.xlim([0, 4000])
        plt.ylim([0, ylim_max])
        plt.xlabel('Raman shift (cm-1)')
        plt.ylabel('Counts')
        plt.minorticks_on()
        plt.show()

def h5data_fetch_spec(h5data, spectrum_names):
    
    if type(spectrum_names) == str:
        current = h5data[spectrum_names]
        spectrum_x = current.attrs['x']
        spectrum_raman = np.reshape(np.array(current), (-1, np.size(spectrum_x)))

        return RamanSpectrum(spectrum_x, spectrum_raman)
        
    elif type(spectrum_names) == list:
        first = h5data[spectrum_names[0]]
        first_x = first.attrs['x']
        output_RamanSpectrum = RamanSpectrum(first_x, np.empty((0, np.size(first_x))))
        
        for i in range(len(spectrum_names)):
            current = h5data[spectrum_names[i]]
            spectrum_x = current.attrs['x']
            if np.array_equal(output_RamanSpectrum.wn, spectrum_x) == False:
                print('Be careful! x-axis vectors are not consistent')
            
            current_raman = np.reshape(np.array(current), (-1, np.size(spectrum_x)))
            output_RamanSpectrum.spec_data = np.vstack((output_RamanSpectrum.spec_data, current_raman))
        
        return output_RamanSpectrum

# def from_h5_create_dict(h5data, create_times_lst=False):
#     f_loaded = h5data['All Raw']
#     spectra_names_lst = list(f_loaded.keys())
#     num_spectra = len(spectra_names_lst)
#     output_dict = {}
#     times_lst = []
#     for i in range(num_spectra):
#         current_spectra_name = spectra_names_lst[i]
#         current_data = f_loaded[current_spectra_name]
#         current_data_raman = np.transpose(np.array(current_data))
#         current_data_x = current_data.attrs['x']
#         current_data_attrs = current_data.attrs

#         keyname = 'spec_'+str(i)
#         output_dict[keyname] = h5data_fetch_spec(f_loaded, current_spectra_name)
#         times_lst.append(current_data_attrs['creation_time'])
#     if create_times_lst:
#         return output_dict, times_lst
#     return output_dict


#########################################

#########################################
### Functions to go from dictionary to h5

def from_dict_write_h5(input_dict, fileloc):
    with h5py.File(fileloc, 'w') as f:
        from_dict_write_h5_helper(input_dict, f)

def from_dict_write_h5_helper(input_dict, h5obj):
    for key,value in input_dict.items():
        if type(value) == RamanSpectrum:
            h5obj[key] = value.spec_data
            h5obj[key].attrs['x'] = value.wn
            h5obj[key].attrs['wavelengths'] = value.wn
            attrs_helper(value.attrs, h5obj[key])
        elif type(value) == dict:
            h5obj.create_group(key)
            from_dict_write_h5_helper(value, h5obj[key])

def attrs_helper(attrs_dict, h5dataset):
    for key, value in attrs_dict.items():
        if key not in ['x', 'wavelengths']:
            h5dataset.attrs[key] = value

#########################################

#########################################
### Functions to analyze dictionaries of raman spectra

def removeCRs_from_dict(input_dict, threshold=5, window_size=5):
    output_dict = {}
    for key in input_dict.keys():
        current = input_dict[key]
        output_dict[key] = current.removeCRs(threshold, window_size)
    return output_dict

def removeCRs_from_dict_charlie(input_dict, reference=1, factor=15):
    output_dict = {}
    for key in input_dict.keys():
        current = input_dict[key]
        output_dict[key] = current.removeCRs_charlie(reference, factor)
    return output_dict

def from_dict_compute_means(input_dict):
    output_dict = {}
    for key in input_dict.keys():
        current = input_dict[key]
        output_dict[key] = RamanSpectrum(current.wn, np.mean(current.spec_data, 0))
    return output_dict

def from_dict_median3_mean(input_dict):
    output_dict = {}
    for key in input_dict.keys():
        current = input_dict[key]
        output_dict[key] = current.median_contract_by_3()

    return from_dict_compute_means(output_dict)

def from_dict_medfilt_mean(input_dict):
    output_dict = {}
    for key in input_dict.keys():
        current = input_dict[key]
        output_dict[key] = current.medfilt_by_acquisition()

    return from_dict_compute_means(output_dict)


def combine_RamanSpectra(input_dict, list_of_keys):
    spec_data_list = []
    for key in list_of_keys:
        wn = input_dict[key].wn
        spec_data_list.append(input_dict[key].spec_data)
    spec_data_list_tuples = tuple(spec_data_list)
    return RamanSpectrum(wn, np.vstack(spec_data_list_tuples))

#########################################

#########################################
### Functions for style purposes

def mpl_inline():
    
    mpl.rcParams['figure.dpi']= 150
    plt.rcParams['figure.facecolor'] = 'white'

def simpleline_style(xlim=[0,4000], ylim=[0, 10000]):
    fig = plt.gcf()
    ax = plt.gca()

    ax.set_xlabel('Raman shift (cm$\\mathregular{^{-1}}$)', )
    ax.set_ylabel('Counts')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.minorticks_on()

    ax.tick_params(direction='in')

    fig.set_size_inches(6, 4)
    
    plt.style.use('ayh_simplelines')

#########################################

#########################################
### Miscellaneous

def wn_to_wl(wn, excitation_line=633):
    return 1/(-wn*1e-7 + 1/excitation_line)

def calculate_max_after_100wn(x, y):
    if y.ndim == 1:
        return np.amax(y[np.where(x>100)])
    elif y.ndim == 2:
        return np.amax(y[:, np.where(x>100)])

def modified_z_score(intensity):
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return abs(modified_z_scores)

def removeCRArray(wn, array, threshold=5, window_size=5, start_wn=100):
    start_wn_idx = min(wn, key=lambda x:abs(x-start_wn))
    
    current_z_score = modified_z_score(array)
    spikes = current_z_score > threshold
    
    array_spikes_rem = copy.copy(array)
    m=window_size
    for i in range(np.size(array)):
        if spikes[i] != 0 and i>start_wn_idx:
            # print(i)
            leftbound = max(i-m, 0)
            rightbound = min(i+m+1, np.size(array))
            w = np.arange(leftbound, rightbound)
            w2 = w[spikes[w] == 0]
            array_spikes_rem[i] = np.mean(array[w2])
    return array_spikes_rem
    
#########################################