from lmfit.models import QuadraticModel, GaussianModel, PolynomialModel
from lmfit import Model
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

def bg_sub(raman_spectra, plot_each=False, reduce_region=[3090,4000], cut_region=[3150,3722], order=2, to_run='all'):
    bg = PolynomialModel(order)
    
    flag=False
    if type(raman_spectra) == AHR.RamanSpectrum:
        raman_spectra = {'a': raman_spectra}
        flag=True
        
    BGsub = {}
    for key, spec in raman_spectra.items():
        if to_run != 'all':
            if key not in to_run:
                continue
        spec_red = spec.reduce_wn_region(reduce_region)
        spec_cut = spec_red.cut_wn_region(cut_region)
        spec_cut_x = spec_cut.wn
        spec_cut_y = np.squeeze(spec_cut.spec_data.T)

        bg_params = bg.make_params(c0=0, c1=0, c2=0, c3=0, c4=0, c5=5, c6=0, c7=0)
        bg_fit = bg.fit(spec_cut_y, x=spec_cut_x, params=bg_params)

        spec_red_bg_sub = AHR.RamanSpectrum(spec_red.wn, np.squeeze(spec_red.spec_data.T) - bg_fit.eval(x=spec_red.wn))

        BGsub[key] = spec_red_bg_sub
        
        if plot_each:
        
            fig, ax = plt.subplots()
            ax.plot(spec_red.wn, spec_red.spec_data.T)
            ax.scatter(spec_cut.wn, spec_cut.spec_data.T, s=3, color='orange')
            ax.plot(spec_red.wn, bg_fit.eval(x=spec_red.wn))
            ax.set_title(key)
    if flag:
        return BGsub['a']
    else:
        return BGsub

def sph(model, param, value, set_min=-np.inf, set_max=np.inf, win=None, v=True):
    if param[-5:] == 'sigma':
        param = param[:-5]+'fwhm'
        value*=2.355
        set_min*=2.355
        set_max*=2.355
        if win != None:
            win*=2.355
    if win==None:
        model.set_param_hint(param, value=value, min=set_min, max=set_max, vary=v)
    else:
        if win==0:
            model.set_param_hint(param, value=value, vary=False)
        else:
            model.set_param_hint(param, value=value, min=value-win, max=value+win, vary=v)
            
def short_fit_report(fit_result, prefixes):
    total_report=''
    for i in np.arange(len(prefixes)):
        prefix_i = prefixes[i]
        center = str(round(fit_result.params[prefix_i+'_center'].value, 2))
        center_vary = '' if fit_result.params[prefix_i+'_center'].vary else '(f)'
        fwhm = str(round(fit_result.params[prefix_i+'_fwhm'].value, 2))
        fwhm_vary = '' if fit_result.params[prefix_i+'_fwhm'].vary else '(f) '
        height = str(round(fit_result.params[prefix_i+'_height'].value, 2))
        result = (prefix_i + ': ' + '(' + center + center_vary + ', ' + fwhm + fwhm_vary + ', ' + height + ')')
        total_report+=(result+'\n')
    return total_report

def gaussian(x, center, height, fwhm):
    sigma = fwhm/2.35482
    return np.abs(height)*np.exp(-np.power((x-center), 2)/(2*sigma**2))
def lorentzian(x, center, height, fwhm):
    width=fwhm/2
    return height / (1 + ((x-center)/width)**2)
def fit_routine(BGsub_spectra, run_gaussians_params_func, to_run='all', 
                piggyback=False, calc_ci='no', return_full_fitresults=False, scale_covar=True,
               print_short_fit_report=True, plot_all=True):
    """
    BGsub_spectra: {expt_time: RamanSpectrum}
    run_gaussians_params_func: returns (model, prefixes) for input: expt_time
    calc_ci: either 'no' or 'yes_fill_stderr' or 'yes'
    returns: fit_results = {expt_time: fit_result_params}
    
    """
    
    print(BGsub_spectra.keys())
    fitresult_dict = {}
    full_fitresult_dict = {}
    colors = get_colors()
    
    last_params = None 
    for expt_time, spec in BGsub_spectra.items():
        
        if (to_run != 'all'):
            if (expt_time not in to_run):
                continue
        
    
        (model, prefixes) = run_gaussians_params_func(expt_time)
        
        params = model.make_params()
        
        if piggyback and (last_params != None):
            for last_param in last_params:
                if last_param in params:
                    last_params_value = last_params[last_param].value
                    epsilon=0.01
                    
                    if last_params_value == params[last_param].min:
                        last_params_value += epsilon
                    elif last_params_value == params[last_param].max:
                        last_params_value -= epsilon
                        
#                     if last_param[-6:] == 'height':
#                         last_params_value = 0.95*last_params_value
                    params[last_param].value = last_params_value
#                     if params[last_param].vary:
#                         params[last_param].value = last_params_value
        
        fitresult = model.fit(np.squeeze(spec.spec_data.T),
                       x=spec.wn,
                       params=params,
                             scale_covar=scale_covar)
        
        if calc_ci == 'yes_fill_stderr':
            for p in fitresult.params:
                fitresult.params[p].stderr = abs(fitresult.params[p].value * 0.1)
            fitresult.conf_interval(fwhms=[1])
        elif calc_ci == 'yes':
            fitresult.conf_interval(fwhms=[1])
        
        fitresult_ci_out = fitresult.ci_out
        last_params=fitresult.params
        
        integrated_area_fitted = np.trapz(fitresult.eval(x=spec.wn), x=spec.wn)
        integrated_area = np.trapz(np.squeeze(spec.spec_data.T), x=spec.wn)

        fitresult_dict[expt_time] = {'params': fitresult.params,
                                     'area': integrated_area,
                                    'area_fitted': integrated_area_fitted,
                                    'ci_out': fitresult_ci_out,
                                    'covar': fitresult.covar,
                                    'fit_report': fitresult.fit_report()}
        
        if return_full_fitresults:
            full_fitresult_dict[expt_time] = fitresult
        
        cmpts = fitresult.eval_components()
        
        if plot_all:
            print(expt_time)
            plt.figure()
            plt.plot(spec.wn, spec.spec_data.T)
            plt.plot(spec.wn, fitresult.eval())

            for cmpt in cmpts:
                plt.title(expt_time)
                if cmpt == 'polynomial':
                    color='black'
                else:
                    color=colors[cmpt[:-1]]
                plt.plot(spec.wn, cmpts[cmpt], color=color)

            if print_short_fit_report:
                fig=plt.gcf()
                fig.set_size_inches(12/2.54, 5.33/2.54)

                fit_annotation = ('Gaussian: (ν0, fwhm, height)\n'+short_fit_report(fitresult, prefixes)+
                                  'Integrated area: '+str("{:.2e}".format(integrated_area)))
                plt.subplots_adjust(left=0.1, bottom=0.1, right=0.5, top=0.9)
                plt.annotate(fit_annotation, (0.51,0.1), xycoords='figure fraction', fontsize=11)

            plt.show();
    
    if return_full_fitresults:
        return fitresult_dict, full_fitresult_dict
    else:
        return fitresult_dict

from matplotlib.lines import Line2D
def get_param_values(dct, times_lst, prefix):
    centers = np.array([dct[expt_time]['params'][prefix+'_center'] for expt_time in times_lst])
    centers_stderrs = np.array([dct[expt_time]['params'][prefix+'_center'].stderr for expt_time in times_lst],
                              dtype=np.float)
    
    fwhms = np.abs(np.array([dct[expt_time]['params'][prefix+'_fwhm'] for expt_time in times_lst]))
    fwhms_stderrs = np.array([dct[expt_time]['params'][prefix+'_fwhm'].stderr for expt_time in times_lst],
                             dtype=np.float)
    
    heights = np.abs(np.array([dct[expt_time]['params'][prefix+'_height'] for expt_time in times_lst]))
    heights_stderrs = np.array([dct[expt_time]['params'][prefix+'_height'].stderr for expt_time in times_lst],
                              dtype=np.float)
    
    areas = np.sqrt(2*np.pi)*fwhms/2.355*heights
    areas_stderrs = areas/2.355*np.sqrt((fwhms_stderrs/fwhms)**2 + (heights_stderrs/heights)**2)
    
    avgs = np.array([round(np.mean(centers),3), 
                    round(np.mean(fwhms),3),
                    round(np.mean(heights),3),
                    round(np.mean(areas),3)])
    centers_stderrs_cleaned = centers_stderrs[~np.isnan(centers_stderrs)]
    centers_stderrs_cleaned = centers_stderrs_cleaned[centers_stderrs_cleaned != 0]
    
    avgs_stderrs = np.array([round( (np.sum(centers_stderrs_cleaned**2))/np.size(centers_stderrs_cleaned),3),
                             None,
                             None,
                             None])
    
    avgs_std_in_mean = np.array([round(np.std(centers_stderrs_cleaned)/np.sqrt(np.size(centers_stderrs_cleaned)),3),
                             None,
                             None,
                             None])
    return {
        'centers': centers,
        'centers_stderrs': centers_stderrs,
        
        'fwhms': fwhms,
        'fwhms_stderrs': fwhms_stderrs,
        
        'heights': heights,
        'heights_stderrs': heights_stderrs,
        
        'areas': areas,
        'areas_stderrs': areas_stderrs,
        
        'avgs': avgs,
        'avgs_stderrs': avgs_stderrs,
        'avgs_std_in_mean': avgs_std_in_mean
    }

def get_param_bounds(dct, times_lst, prefix):
    centers_mins = np.array([dct[expt_time]['params'][prefix+'_center'].min for expt_time in times_lst])
    centers_maxs = np.array([dct[expt_time]['params'][prefix+'_center'].max for expt_time in times_lst])
    
    if not (np.all(centers_mins == centers_mins[0]) & np.all(centers_maxs == centers_maxs[0])):
        print('Warning: centers bds not all same!')
    centers_bds = (centers_mins, centers_maxs)
    
    fwhms_mins = np.array([dct[expt_time]['params'][prefix+'_fwhm'].min for expt_time in times_lst])
    fwhms_maxs = np.array([dct[expt_time]['params'][prefix+'_fwhm'].max for expt_time in times_lst])
    
    if not (np.all(fwhms_mins == fwhms_mins[0]) & np.all(fwhms_maxs == fwhms_maxs[0])):
        print('Warning: fwhms bds not all same!')
    fwhms_bds = (fwhms_mins, fwhms_maxs)
    
    return {
        'centers_bds': centers_bds,
        'fwhms_bds': fwhms_bds,
    }

def get_total_areas(dct, times_lst, prefix_lst):
    all_areas = []
    for prefix in prefix_lst:
        fwhms = np.array([dct[expt_time]['params'][prefix+'_fwhm'] for expt_time in times_lst])
        heights = np.array([dct[expt_time]['params'][prefix+'_height'] for expt_time in times_lst])
        areas = np.sqrt(2*np.pi)*fwhms/2.355*heights
        all_areas.append(areas)
    return sum(all_areas)

def total_integrated_areas(fitresults_dict, times):
    integrated_areas = []
    integrated_areas_fitted = []
    for time in times:
        area = fitresults_dict[time]['area']
        area_fitted = fitresults_dict[time]['area_fitted']
#         integral_spec = np.trapz(np.squeeze(spec.spec_data), spec.wn)
        integrated_areas.append(area)
        integrated_areas_fitted.append(area_fitted)
    return (integrated_areas, integrated_areas_fitted)
        
def plot_param_results(fitresults_dict, times, peak_lst, bg_sub_dict=None, plot_bounds=True):
    tab10 = plt.get_cmap('tab10')
    colors = get_colors()
    
    peak_results = []
    peak_bds = []
    for peak in peak_lst:
        peak_results.append(get_param_values(fitresults_dict, times, peak))
        peak_bds.append(get_param_bounds(fitresults_dict, times, peak))
    
    ### Begin: plot central frequencies ###
    plt.figure()
    custom_lines = []
    for (i, peak_result) in enumerate(peak_results):
        plt.errorbar(times, peak_result['centers'], yerr=peak_result['centers_stderrs'], 
                     fmt='-', markersize=3, linewidth=1, color=colors[peak_lst[i]])
        custom_lines.append(Line2D([0], [0], color=colors[peak_lst[i]], lw=2))
        if plot_bounds:
            plt.plot(times, peak_bds[i]['centers_bds'][0], '-', color=colors[peak_lst[i]], linewidth=0.5)
            plt.plot(times, peak_bds[i]['centers_bds'][1], '-', color=colors[peak_lst[i]], linewidth=0.5)
    AHR.simpleline_style(xlim=None, ylim=None)
    
    plt.xlabel('Time after expt start (min)')
    plt.ylabel('Peak center freq')
    
    plt.legend(custom_lines, peak_lst, ncol=2, fontsize='small')
    plt.title('Peak center freqs')
    plt.xlim([0, 220])
    plt.show()
    ### End: plot central frequencies ###
    
    ### Begin: plot fwhms ###
    plt.figure()
    custom_lines = []
    for (i, peak_result) in enumerate(peak_results):
        plt.errorbar(times, peak_result['fwhms'], yerr=peak_result['fwhms_stderrs'],
                     fmt='-', markersize=3, linewidth=1, color=colors[peak_lst[i]])
        custom_lines.append(Line2D([0], [0], color=colors[peak_lst[i]], lw=2))
        if plot_bounds:
            plt.plot(times, peak_bds[i]['fwhms_bds'][0], '-', color=colors[peak_lst[i]], linewidth=0.5)
            plt.plot(times, peak_bds[i]['fwhms_bds'][1], '-', color=colors[peak_lst[i]], linewidth=0.5)
    AHR.simpleline_style(xlim=None, ylim=None)
    
    plt.xlabel('Time after expt start (min)')
    plt.ylabel('Peak fwhm')
    
    plt.legend(custom_lines, peak_lst, ncol=2, fontsize='small')
    plt.title('Peak fwhm')
    plt.xlim([0, 220])
    plt.show()
    ### End: plot plot fwhms ###
    
    ### Begin: plot heights ###
    plt.figure()
    custom_lines = []
    for (i, peak_result) in enumerate(peak_results):
        plt.errorbar(times, peak_result['heights'], yerr=peak_result['heights_stderrs'],
                 fmt='-', markersize=3, linewidth=1, color=colors[peak_lst[i]])
        custom_lines.append(Line2D([0], [0], color=colors[peak_lst[i]], lw=2))
    AHR.simpleline_style(xlim=None, ylim=None)
    
    plt.xlabel('Time after expt start (min)')
    plt.ylabel('Peak heights')
    
    plt.legend(custom_lines, peak_lst, ncol=2, fontsize='small')
    plt.title('Peak height')
    plt.xlim([0, 220])
    plt.show()
    ### End: plot plot heights ###
    
    ### Begin: plot areas ###
    plt.figure()
    custom_lines = []
    for (i, peak_result) in enumerate(peak_results):
        plt.errorbar(times, peak_result['areas'], yerr=peak_result['areas_stderrs'],
                 fmt='-', markersize=3, linewidth=1, color=colors[peak_lst[i]])
        custom_lines.append(Line2D([0], [0], color=colors[peak_lst[i]], lw=2))
    AHR.simpleline_style(xlim=None, ylim=None)
    
    plt.xlabel('Time after expt start (min)')
    plt.ylabel('Peak areas')
    
    plt.legend(custom_lines, peak_lst, ncol=2, fontsize='small')
    plt.title('Peak areas')
    plt.xlim([0, 220])
    plt.show()
    ### End: plot plot areas ###
    
    if bg_sub_dict != None:
        integrated_areas = total_integrated_areas(fitresults_dict, times)
        integrated_areas_normal = integrated_areas[0]
        integrated_areas_fitted = integrated_areas[1]
        
        ### Begin: plot integrated areas ###
        plt.figure()
        plt.plot(times, integrated_areas_normal, '-', linewidth=1, markersize=3, color='black')
        plt.title('Integrated band area')
        plt.xlabel('Time after expt start (min)')
        plt.ylabel('Area')
        AHR.simpleline_style(xlim=None, ylim=None)
        plt.xlim([0, 220])
        plt.show()
        
        plt.figure()
        plt.plot(times, integrated_areas_fitted, '-', linewidth=1, markersize=3, color='black')
        plt.title('Integrated band area (fitted curves)')
        plt.xlabel('Time after expt start (min)')
        plt.ylabel('Area of fitted curves')
        AHR.simpleline_style(xlim=None, ylim=None)
        plt.xlim([0, 220])
        plt.show()
        ### End: plot integrated areas ###
        
        ### Begin: plot normalized areas ###
        
        plt.figure()
        custom_lines = []
        for (i, peak_result) in enumerate(peak_results):
            normalized_areas = peak_result['areas']/integrated_areas_normal
            normalized_areas_stderrs = peak_result['areas_stderrs']/integrated_areas_normal
            plt.errorbar(times, normalized_areas, yerr=normalized_areas_stderrs,
                         fmt='-', markersize=3, linewidth=1, color=colors[peak_lst[i]])
            custom_lines.append(Line2D([0], [0], color=colors[peak_lst[i]], lw=2))
        AHR.simpleline_style(xlim=None, ylim=None)

        plt.xlabel('Time after expt start (min)')
        plt.ylabel('Peak areas')

        plt.legend(custom_lines, peak_lst, ncol=2, fontsize='small')
        plt.title('Normalized peak areas')
#         plt.yscale('log'); plt.ylim([1e-3, 0.11])
        plt.xlim([0, 220])
        plt.show()
        
        plt.figure()
        custom_lines = []
        for (i, peak_result) in enumerate(peak_results):
            normalized_areas_fitted = peak_result['areas']/integrated_areas_fitted
            normalized_areas_fitted_stderrs = peak_result['areas_stderrs']/integrated_areas_fitted
            plt.errorbar(times, normalized_areas_fitted, yerr=normalized_areas_fitted_stderrs,
                     fmt='-', markersize=3, linewidth=1, color=colors[peak_lst[i]])
            custom_lines.append(Line2D([0], [0], color=colors[peak_lst[i]], lw=2))
        AHR.simpleline_style(xlim=None, ylim=None)

        plt.xlabel('Time after expt start (min)')
        plt.ylabel('Peak areas')

        plt.legend(custom_lines, peak_lst, ncol=2, fontsize='small')
        plt.title('Normalized (fitted) peak areas')
#         plt.yscale('log'); plt.ylim([1e-3, 0.11])
        plt.xlim([0, 220])
        plt.show()
        ### End: plot plot normalized areas ###
        
def extract_peak_results(fitresults_dict, times, peak_lst):
    peak_results = []
    peak_bds = []
    integrated_areas = np.array([fitresults_dict[expt_time]['area'] for expt_time in times])
    for peak in peak_lst:
        gpv_result = get_param_values(fitresults_dict, times, peak)
        normalized_dict = {
            'areas_norm': gpv_result['areas']/integrated_areas,
            'areas_norm_stderrs': gpv_result['areas_stderrs']/integrated_areas,
        }
        gpv_result_with_normalized_areas = {**normalized_dict, **gpv_result}
        
        peak_results.append(gpv_result_with_normalized_areas)
        peak_bds.append(get_param_bounds(fitresults_dict, times, peak))    
    
    return (peak_results, peak_bds)

def full_summary_figure_plot_helper(ax, times, peak_results, result_name, peak_lst,
                                    ylabel, legend, 
                                    plt_fmt, plt_markersize, plt_lw, colors):
    custom_lines = []
    for i, peak_result in enumerate(peak_results):
        
        ax.errorbar(times, peak_result[result_name], yerr=peak_result[result_name+'_stderrs'], 
                        fmt=plt_fmt, markersize=plt_markersize, linewidth=plt_lw, color=colors[peak_lst[i]])
        custom_lines.append(Line2D([0], [0], color=colors[peak_lst[i]], lw=2))
    ax.set_ylabel(ylabel)
    ax.set_xlim([-20, 360])
    ax.minorticks_on()
    ax.legend(custom_lines, legend, ncol=2, fontsize='x-small')

def get_colors():
    tab10 = plt.get_cmap('tab10')
    tab20 = plt.get_cmap('tab20')
    pastel1 = plt.get_cmap('Pastel1')
    set3 = plt.get_cmap('Set3')
    colors = {
        'OD_x': tab10(0),
        'OH_x': tab10(0),
        'OD_y': tab10(1),
        'OH_y': tab10(1),
        'OD_z': tab10(2),
        'OH_z': tab10(2),
        'OD_xxx': tab10(3),
        'OD_yyy': tab10(4),
        'OD_yz': tab10(5),
        'OD_A': tab10(6),
        'OH_A': tab10(6),
        'OD_B': tab10(7),
        'OH_B': tab10(7),
        'OD_C': tab10(8),
        'OH_C': tab10(8),
        'OD_D': tab10(9),
        'OH_D': tab10(9),
        'OH_xy': pastel1(0),
        'p830': set3(0),
        'p450': set3(3),
        'OD_w': set3(6),
        'p1660': tab10(8),
        'p1690': set3(2),
        'p1750': set3(4),
        'p1780': set3(7),
        'OD_l': set3(9),
        'OD_m': set3(10),
        'OD_o': set3(5),
        'OD_p': set3(8),
        'p2320': tab20(9),
        'p2330': tab20(8)
    }
    return colors
