from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import subprocess
import sys
import os
# Add the parent directory of 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.plotting import *
from utils.plotting import cfg as cfg_plt
import utils.plotting as plotting
import matplotlib.gridspec as gridspec
from matplotlib import pyplot, image, transforms
import numpy as np
import h5py
import matplotlib.pyplot as plt
import importlib
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from scipy.stats import wasserstein_distance
import random 
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close()
mpl.rcParams['xtick.labelsize'] = 18   
mpl.rcParams['ytick.labelsize'] = 18

mpl.rcParams['font.size'] = 20
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['text.usetex'] = False #True
mpl.rcParams['font.family'] = 'serif'

# name_folder = 'HGx4_1SF_ECAL_HCAL_LogE!_50'
name_folder = 'HGx9_CNF_ECAL_HCAL_30'
shw = 5000 
font = 25 #25
Ymin, Ymin_hcal, Ymax = 0, 30, 78
#save arrays 
save_dir_fake_15 = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/'+name_folder+'/15GeV_'+str(shw)
save_dir_fake_50 = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/'+name_folder+'/50GeV_'+str(shw) 
save_dir_fake_85 = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/'+name_folder+'/85GeV_'+str(shw)  

save_dir_real_15 = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/Geant4/15GeV_'+str(shw)  
save_dir_real_50 = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/Geant4/50GeV_'+str(shw)  
save_dir_real_85 = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/Geant4/85GeV_'+str(shw)   
print('Getting features...') # plots for paper/presentation
f_names=['e_radial','e_radial_ecal', 'e_radial_hcal', 'occ_list', 'occ_list_025', 'e_sum_list', 
             'e_sum_ecal_list', 'e_sum_hcal_list', 'hits_list', 'hits_ecal_list', 'hits_hcal_list', 'e_layers_list', 
             'e_layers_ecal_list', 'e_layers_hcal_list',  'X', 'Z', 'numb_active_hits_list', 'numb_active_hits_ecal', 
             'numb_active_hits_hcal', 'start_layer_list','cog_x', 'cog_y', 'cog_z']
f_names_r=['e_radial_r','e_radial_ecal_r', 'e_radial_hcal_r', 'occ_list_r', 'occ_list_025_r', 'e_sum_list_r', 
             'e_sum_ecal_list_r', 'e_sum_hcal_list_r', 'hits_list_r', 'hits_ecal_list_r', 'hits_hcal_list_r', 'e_layers_list_r',
             'e_layers_ecal_list_r', 'e_layers_hcal_list_r', 'X_r', 'Z_r', 'numb_active_hits_list_r', 'numb_active_hits_ecal_r', 
             'numb_active_hits_hcal_r', 'start_layer_list_r', 'cog_x_r', 'cog_y_r', 'cog_z_r']

print('LOADING...')
real_dict_15, real_dict_50, real_dict_85 = {}, {}, {}
for k, name in enumerate(f_names_r): 
    real_dict_15[name] = np.load(save_dir_real_15+'/'+name+'.npy')
    real_dict_50[name] = np.load(save_dir_real_50+'/'+name+'.npy')
    real_dict_85[name] = np.load(save_dir_real_85+'/'+name+'.npy')
    if k==0: print(real_dict_85[name][0].shape)
        
fake_dict_15, fake_dict_50, fake_dict_85  = {}, {}, {}
for k, name in enumerate(f_names): 
    fake_dict_15[name] = np.load(save_dir_fake_15+'/'+name+'.npy')
    fake_dict_50[name] = np.load(save_dir_fake_50+'/'+name+'.npy')   
    fake_dict_85[name] = np.load(save_dir_fake_85+'/'+name+'.npy')

title_list = ['ECAL', 'HCAL', '']
col = ['#4daf4a', '#ff7f00', '#984ea3'] #colorblind colors

def compute_core90(data):
    """
    Computes µ90 (mean) and σ90 (RMS) for the 90% core of each distribution.

    Parameters:
        distributions (dict): A dictionary where keys are energy levels and
                              values are numpy arrays of distribution data.

    Returns:
        dict: A dictionary with energy as key and a tuple (mu90, sigma90) as value.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    lower_idx = int(n * 0.05)
    upper_idx = int(n * 0.95)
    core_90 = sorted_data[lower_idx:upper_idx]
    return core_90

def error_sigma_over_mean(x): 
    
    def sigma_error(x):
        n = x.shape[0]
        mean = np.mean(x)
        sigma = np.std(x, ddof=1)
        mu4 = np.sum((x-mean)**4 * (1/n)) # fourth moment
        return 1/(2*sigma) * np.sqrt(1/n * (mu4 - (n-3)/(n-1) * sigma**4)  )
     
    sigma_err = sigma_error(x) 
    mean_err = np.std(x, ddof=1)/ np.sqrt(x.shape[0]) 
    return np.sqrt(sigma_err**2 / np.mean(x)**2 + mean_err**2 * sigma_err**2 / np.mean(x)**4)
    
    
def plotting_energy_sum(my_dir):
    fig = plt.figure(10, figsize=(24,8))
    gs = gridspec.GridSpec(1, 3)
    to_plot_r_list = ["e_sum_ecal_list_r", "e_sum_hcal_list_r", "e_sum_list_r"]
    to_plot_list = ["e_sum_ecal_list", "e_sum_hcal_list", "e_sum_list"]
    
    for j in range(3): 
        axs = [plt.subplot(gs[j])]
        to_plot_r, to_plot = to_plot_r_list[j], to_plot_list[j]
        _range=(np.nanmin([real_dict_15[to_plot_r], fake_dict_15[to_plot]]), np.nanmax([real_dict_85[to_plot_r], fake_dict_85[to_plot]]))
        if j==0: _range=(50, 2500)
        else: _range=(np.nanmin([real_dict_15[to_plot_r], fake_dict_15[to_plot]]), np.nanmax([real_dict_85[to_plot_r]])) 
        
        h_data = axs[0].hist(np.array(real_dict_15[to_plot_r]), bins = 40, range=_range, alpha = 0.2, linewidth=1.6, color=col[0])
        h_gen = axs[0].hist(np.array(fake_dict_15[to_plot]), bins = 40, range=_range, histtype='step', linewidth=4, linestyle ='-', color=col[0], label='15GeV')
    
        h_data = axs[0].hist(np.array(real_dict_50[to_plot_r]), bins = 40, range=_range, alpha = 0.2, linewidth=1.6, color=col[1])
        h_gen = axs[0].hist(np.array(fake_dict_50[to_plot]), bins = 40, range=_range, histtype='step', linewidth=4, linestyle ='-', color=col[1], label='50GeV')
        
        h_data = axs[0].hist(np.array(real_dict_85[to_plot_r]), bins = 40, range=_range, alpha = 0.2, linewidth=1.6, color=col[2])
        h_gen = axs[0].hist(np.array(fake_dict_85[to_plot]), bins = 40, range=_range, histtype='step', linewidth=4, linestyle ='-', color=col[2], label='85GeV')
        
        pgeant = axs[0].hist([0,0], bins=1, label='geant4', color='gray', alpha=0.2)
        pgen = axs[0].hist([0,0], bins=1, label='CaloHadronic', histtype='step', color='gray') 
        axs[0].set_xlim(_range[0], _range[1])         
        axs[0].set_xlabel('Energy Sum '+title_list[j], fontsize=font)
        axs[0].set_ylabel('$\#$ Showers', fontsize=font)
        axs[0].tick_params(axis="x", labelsize=font) 
        axs[0].tick_params(axis="y", labelsize=font)  
        
        h, l = axs[0].get_legend_handles_labels()
        # ax2 = axs[0].twinx()
        if j==0: axs[0].legend([h[3],h[4]],[l[3],l[4]], loc=1, ncols=1, fontsize=font) 
        else: axs[0].legend([h[0],h[1],h[2]],[l[0],l[1],l[2]], loc=1, fontsize=font)
        # ax2.set_axis_off()
        del h, l
    plt.tight_layout()   
    plt.savefig(my_dir+'/EnergySum_'+str(shw)+ '_showers.pdf', dpi=100, bbox_inches='tight')
    plt.close()
    
def plotting_hits(my_dir):
    plt.figure(2, figsize=(24,8))
    gs = gridspec.GridSpec(1, 3)
    to_plot_r_list = ["numb_active_hits_ecal_r", "numb_active_hits_hcal_r", "numb_active_hits_list_r"]
    to_plot_list = ["numb_active_hits_ecal", "numb_active_hits_hcal", "numb_active_hits_list"]
    
    for j in range(3): 
        axs = [plt.subplot(gs[j])]
        to_plot_r, to_plot = to_plot_r_list[j], to_plot_list[j]
        real_dict_15[to_plot_r] = np.array(real_dict_15[to_plot_r]) 
        if j==0: _range=(80, 2500)
        else: _range=(np.nanmin([real_dict_15[to_plot_r], fake_dict_15[to_plot]]), np.nanmax([real_dict_85[to_plot_r]])) 
        
        h_data = axs[0].hist(np.array(real_dict_15[to_plot_r]), bins = 40, range=_range, alpha = 0.2, linewidth=1.6, color=col[0])
        h_gen = axs[0].hist(np.array(fake_dict_15[to_plot]), bins = 40, range=_range, histtype='step', linewidth=4,linestyle ='-', color=col[0], label='15GeV')
    
        h_data = axs[0].hist(np.array(real_dict_50[to_plot_r]), bins = 40, range=_range, alpha = 0.2, linewidth=1.6, color=col[1])
        h_gen = axs[0].hist(np.array(fake_dict_50[to_plot]), bins = 40, range=_range, histtype='step', linewidth=4,linestyle ='-', color=col[1], label='50GeV')
        
        h_data = axs[0].hist(np.array(real_dict_85[to_plot_r]), bins = 40, range=_range, alpha = 0.2, linewidth=1.6, color=col[2])
        h_gen = axs[0].hist(np.array(fake_dict_85[to_plot]), bins = 40, range=_range, histtype='step', linewidth=4,linestyle ='-', color=col[2], label='85GeV')
        
        axs[0].hist([0,0], bins=1, label='geant4', color='gray', alpha=0.2)
        axs[0].hist([0,0], bins=1, label='CaloHadronic', histtype='step', color='gray')  
        axs[0].set_xlim(_range[0], _range[1])         
        axs[0].set_xlabel('# Hits '+title_list[j], fontsize=font)
        axs[0].set_ylabel('$\#$ Showers', fontsize=font)
        axs[0].tick_params(axis="x", labelsize=font)
        axs[0].tick_params(axis="y", labelsize=font)
        
        h, l = axs[0].get_legend_handles_labels()
        # ax2 = axs[0].twinx()
        if j==0: axs[0].legend([h[3],h[4]],[l[3],l[4]], loc=1, ncols=1, fontsize=font) 
        else: axs[0].legend([h[0],h[1],h[2]],[l[0],l[1],l[2]], loc=1, fontsize=font)
        # ax2.set_axis_off()
    plt.tight_layout()   
    plt.savefig(my_dir+'/Hits_'+str(shw)+ '_showers.pdf', dpi=100, bbox_inches='tight')     
    plt.close()   

def res(x):
    # cond = (x > (np.mean(x)-np.std(x))) | (x < (np.mean(x)+np.std(x)))
    b=5
    batch_size = int(x.shape[0]/b)
    t=[]
    for b in range(b):
        t.append(np.std(x[b*batch_size:(b+1)*batch_size], ddof=1))
    
    error_std = np.std(t, ddof=0) / np.sqrt(len(t)) # I do the error of the std as the error of the mean
    error_mean = np.std(x, ddof=0) / np.sqrt(x.shape[0])
    error = np.sqrt( error_std**2 / np.mean(x)**2 + error_mean**2 * error_std**2 / np.mean(x)**4 )
    return np.std(x, ddof=1) / np.mean(x), error
    
def plotting_energy_resolution_linearity(my_dir):
    marker_size = 15
    inc_en = [15, 50, 85] 
    fig = plt.figure(figsize=(24,19)) 
    # gs = gridspec.GridSpec(1, 4)
    to_plot_r_list = [["numb_active_hits_ecal_r", "numb_active_hits_hcal_r"], 
                      ["e_sum_ecal_list_r", "e_sum_hcal_list_r"]]
    to_plot_list = [["numb_active_hits_ecal", "numb_active_hits_hcal"], 
                    ["e_sum_ecal_list", "e_sum_hcal_list"]]
    
    title_list = ['Number of Hits', 'Total visible energy'] 
    labb = ['ECal', 'HCal']
    subfigs = fig.subfigures(2, 1) #, wspace=0.15) #, wspace=0.07)
    cap_size = 12 
    
    for j in range(2):
        axs = subfigs[j].subplots(1, 2)
        # axs = [subfigs[j].subplot(gs[]), subfigs[j].subplot(gs[j*2+1])]
        for jj in range(2):
            to_plot_r, to_plot = to_plot_r_list[j][jj], to_plot_list[j][jj]
            y_r = [np.array(real_dict_15[to_plot_r]),
                    np.array(real_dict_50[to_plot_r]),
                    np.array(real_dict_85[to_plot_r])]
            y_g = [np.array(fake_dict_15[to_plot]),
                    np.array(fake_dict_50[to_plot]),
                    np.array(fake_dict_85[to_plot])]

            h_data = axs[0].errorbar(inc_en, [res(y)[0] for y in y_r], yerr=[error_sigma_over_mean(y) for y in y_r],  fmt='-o',capsize=cap_size, color=col[jj+1], markersize=marker_size, label=labb[jj])
            h_gen = axs[0].errorbar(inc_en, [res(y)[0] for y in y_g], yerr=[error_sigma_over_mean(y) for y in y_g], fmt='-*', capsize=cap_size, color=col[jj+1], markersize=marker_size, markeredgecolor='black', ecolor='k') 
            
            h_data = axs[1].errorbar(inc_en, [np.mean(y) for y in y_r], yerr=[np.std(y)/np.sqrt(y.shape[0]) for y in y_r], fmt='-o', capsize=cap_size, color=col[jj+1], markersize=marker_size, label=labb[jj])
            h_gen = axs[1].errorbar(inc_en, [np.mean(y) for y in y_g], yerr=[np.std(y)/np.sqrt(y.shape[0]) for y in y_g], fmt='-*', capsize=cap_size, color=col[jj+1], markersize=marker_size, markeredgecolor='black', ecolor='k')
        
        subfigs[j].suptitle(title_list[j], fontsize=font+7, fontweight="bold")    
        axs[0].set_xlabel('Incident Energy [GeV]', fontsize=font+3)
        axs[1].set_xlabel('Incident Energy [GeV]', fontsize=font+3)
        axs[0].set_ylabel('$\sigma$ / $\mu$ ', fontsize=font+3)
        if j==1: axs[1].set_ylabel(' $\mu$ [MeV] ', fontsize=font+3)
        else: axs[1].set_ylabel(' $\mu$ ', fontsize=font+3)
        # axs[1].yaxis.set_label_position("right")
        # axs[1].yaxis.tick_right()
        axs[0].set_ylim(0, 2)
        axs[1].set_ylim(0, 1500)
            
        axs[0].tick_params(axis="x", labelsize=font) 
        axs[0].tick_params(axis="y", labelsize=font) 
        axs[1].tick_params(axis="x", labelsize=font) 
        axs[1].tick_params(axis="y", labelsize=font)   
        axs[1].errorbar(0, -100, 1, fmt='-o',  markersize=marker_size, capsize=cap_size, label='Geant4', color=col[jj+1])
        axs[1].errorbar(0, -100, 1, fmt='-*', markersize=marker_size, capsize=cap_size, label='CaloHadronic', color=col[jj+1], markeredgecolor='black', ecolor='k') 
         
        h, l = axs[0].get_legend_handles_labels()
        h1, l1 = axs[1].get_legend_handles_labels()
        print(l, l1)
        # ax2 = axs[0].twinx()
        if j==0: axs[1].legend([h1[2],h1[3]],[l1[2],l1[3]], loc=2, ncols=1, fontsize=font+3) 
        if j==1: axs[0].legend([h[0],h[1]],[l[0],l[1]], loc=2, ncols=1, fontsize=font+3)
        # else: axs[0].legend([h[0],h[1],h[2]],[l[0],l[1],l[2]], loc=2, fontsize=font)
        # ax2.set_axis_off()
        del h, l
    # plt.tight_layout()   
    print('Saving figures...')
    plt.savefig(my_dir+'/EnergyResolution_'+str(shw)+ '_showers.pdf', dpi=100, bbox_inches='tight')
    plt.close()

def plotting_energy_resolution_linearity_all(my_dir):
    marker_size = 15
    inc_en = [15, 50, 85] 
    fig = plt.figure(figsize=(24,19)) 
    # gs = gridspec.GridSpec(1, 4)
    to_plot_r_list = [["numb_active_hits_list_r"], 
                      ["e_sum_list_r"]]
    to_plot_list = [["numb_active_hits_list"], 
                    ["e_sum_list"]]
    
    title_list = ['Number of Hits', 'Total visible energy'] 
    labb = ['CaloHadronic']
    subfigs = fig.subfigures(2, 1) #, wspace=-0.15) #, wspace=0.07)
    cap_size = 12
    
    for j in range(2):
        axs = subfigs[j].subplots(1, 2)
        # axs = [subfigs[j].subplot(gs[]), subfigs[j].subplot(gs[j*2+1])]
        for jj in range(1):
            to_plot_r, to_plot = to_plot_r_list[j][jj], to_plot_list[j][jj]
            y_r = [compute_core90(np.array(real_dict_15[to_plot_r])),
                    compute_core90(np.array(real_dict_50[to_plot_r])),
                    compute_core90(np.array(real_dict_85[to_plot_r]))]
            y_g = [compute_core90(np.array(fake_dict_15[to_plot])),
                    compute_core90(np.array(fake_dict_50[to_plot])),
                    compute_core90(np.array(fake_dict_85[to_plot]))]
            
            print([error_sigma_over_mean(y) for y in y_r])
            print([res(y)[1] for y in y_r]) 
            h_data = axs[0].errorbar(inc_en, [res(y)[0] for y in y_r], yerr=[error_sigma_over_mean(y) for y in y_r],  fmt='-o', capsize=cap_size, color='k', markersize=marker_size, label=labb[jj])
            h_gen = axs[0].errorbar(inc_en, [res(y)[0] for y in y_g], yerr=[error_sigma_over_mean(y) for y in y_g], fmt='-*', capsize=cap_size, color=col[jj+1], markersize=marker_size) 
            
            h_data = axs[1].errorbar(inc_en, [np.mean(y) for y in y_r], yerr=[np.std(y)/np.sqrt(y.shape[0]) for y in y_r], fmt='-o', capsize=cap_size, color='k', markersize=marker_size, label=labb[jj])
            h_gen = axs[1].errorbar(inc_en, [np.mean(y) for y in y_g], yerr=[np.std(y)/np.sqrt(y.shape[0]) for y in y_g], fmt='-*', capsize=cap_size, color=col[jj+1], markersize=marker_size)
        
        subfigs[j].suptitle(title_list[j], fontsize=font+7, fontweight="bold")    
        axs[0].set_xlabel('Incident Energy [GeV]', fontsize=font+3)
        axs[1].set_xlabel('Incident Energy [GeV]', fontsize=font+3)
        axs[0].set_ylabel('$\sigma_{90}$ / $\mu_{90}$ ', fontsize=font+3)
        if j==1: axs[1].set_ylabel(' $\mu_{90}$ [MeV] ', fontsize=font+3)
        else: axs[1].set_ylabel(' $\mu_{90}$ ', fontsize=font+3) 
        # axs[1].yaxis.set_label_position("right")
        # axs[1].yaxis.tick_right()
        axs[0].set_ylim(0, 1)
        axs[0].set_xlim(10, 90)
        axs[1].set_ylim(0, 2000)
        axs[1].set_xlim(10, 90)
        
        axs[0].tick_params(axis="x", labelsize=font) 
        axs[0].tick_params(axis="y", labelsize=font) 
        axs[1].tick_params(axis="x", labelsize=font) 
        axs[1].tick_params(axis="y", labelsize=font)   
        axs[0].errorbar(0, -100, 1, fmt='-*', capsize=cap_size, markersize=marker_size, label='CaloHadronic', color=col[jj+1]) 
        axs[0].errorbar(0, -100, 1, fmt='-o', capsize=cap_size, markersize=marker_size, label='Geant4', color='k')
        
        h, l = axs[0].get_legend_handles_labels()
        h1, l1 = axs[1].get_legend_handles_labels()
        print(l, l1)
        # ax2 = axs[0].twinx()
        if j==1: axs[0].legend([h[1],h[2]],[l[1],l[2]], loc=2, ncols=1, fontsize=font+3) 
        
        del h, l
    # plt.tight_layout()   
    print('Saving figures...')
    plt.savefig(my_dir+'/EnergyResolutionALL_'+str(shw)+ '_showers.pdf', dpi=100, bbox_inches='tight')
    plt.close()
     
def plotting_correlations(my_dir): 
    _range= [0, 1900] 
    thebins = [50, 50]
    cmin = 0
    to_plot_r = ["e_sum_hcal_list_r","e_sum_ecal_list_r"]
    to_plot = ["e_sum_hcal_list","e_sum_ecal_list"]
    
    fig = plt.figure(10, figsize=(24,8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 4, 4, 4]) #, wspace=0.4)
    H_15, _, _ = np.histogram2d(fake_dict_15[to_plot[0]], fake_dict_15[to_plot[1]], bins=thebins, range=[_range, _range])
    H_r_15, _, _ = np.histogram2d(real_dict_15[to_plot_r[0]], real_dict_15[to_plot_r[1]], bins=thebins, range=[_range, _range]) 
    H_50, _, _ = np.histogram2d(fake_dict_50[to_plot[0]], fake_dict_50[to_plot[1]], bins=thebins, range=[_range, _range])
    H_r_50, _, _ = np.histogram2d(real_dict_50[to_plot_r[0]], real_dict_50[to_plot_r[1]], bins=thebins, range=[_range, _range])
    H_85, _, _ = np.histogram2d(fake_dict_85[to_plot[0]], fake_dict_85[to_plot[1]], bins=thebins, range=[_range, _range])
    H_r_85, _, _ = np.histogram2d(real_dict_85[to_plot_r[0]], real_dict_85[to_plot_r[1]], bins=thebins, range=[_range, _range])
    cmax = max(H_15.max(), H_r_15.max(), H_50.max(), H_r_50.max(), H_85.max(), H_r_85.max())
    shared_norm = mpl.colors.LogNorm(vmin=1e0, vmax=cmax)
    
    axs1 = plt.subplot(gs[1])
    hist3 = axs1.hist2d(
        real_dict_85[to_plot_r[0]], real_dict_85[to_plot_r[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=shared_norm, cmap='BuPu',
    )
    hist2 = axs1.hist2d(
        real_dict_50[to_plot_r[0]], real_dict_50[to_plot_r[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=shared_norm, cmap='Oranges',
    )
    hist1 = axs1.hist2d(
        real_dict_15[to_plot_r[0]], real_dict_15[to_plot_r[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, cmap='Greens', norm=shared_norm,
    )
    
    axs1.set_title("Geant4", fontsize=font)
    axs1.set_xlabel("Visible energy HCal [MeV]", fontsize=font)
    axs1.set_ylabel("Visible energy ECal [MeV]", fontsize=font)
    # fig.colorbar(hist1[3], ax=axs1, location='bottom')
    
    axs0 = plt.subplot(gs[0])
    divider = make_axes_locatable(axs0)
    cax1 = divider.append_axes("right", size="33%", pad=0.8)
    cax2 = divider.append_axes("right", size="33%", pad=0.8)
    cax3 = divider.append_axes("right", size="33%", pad=0.8)
    cbar1 = fig.colorbar(hist1[3], cax=cax1)
    cbar2 = fig.colorbar(hist2[3], cax=cax2)
    cbar3 = fig.colorbar(hist3[3], cax=cax3)
    # cbar1.set_ticks([])
    # cbar2.set_ticks([])
    axs0.set_axis_off()
    
    axs2 = plt.subplot(gs[2])
    hist3 = axs2.hist2d(
        fake_dict_85[to_plot[0]], fake_dict_85[to_plot[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=shared_norm, cmap='BuPu',
    )
    hist2 = axs2.hist2d(
        fake_dict_50[to_plot[0]], fake_dict_50[to_plot[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=shared_norm, cmap='Oranges',
    )
    hist1 = axs2.hist2d(
        fake_dict_15[to_plot[0]], fake_dict_15[to_plot[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=shared_norm, cmap='Greens',
    )
    
    axs2.set_title("CaloHadronic", fontsize=font)
    axs2.set_xlabel("Visible energy HCal [MeV]", fontsize=font)
    axs2.set_ylabel("Visible energy ECal [MeV]", fontsize=font)
    # fig.colorbar(hist2[3], ax=axs2, location='bottom') 
    
    axs3 = plt.subplot(gs[3])    

    thr_ecal = 38 # range= [0,1900] - bins=50: 1900/50 = 38
    range_ = [0, 2000]
    bbins= 50
    axs3.hist(real_dict_15[to_plot_r[0]][real_dict_15[to_plot_r[1]]< thr_ecal], 
              bins=bbins, range=range_, color='green', alpha=0.2) #, density=True)
    axs3.hist(fake_dict_15[to_plot[0]][fake_dict_15[to_plot[1]]< thr_ecal], 
              bins=bbins, range=range_, color='green', label='15 GeV', histtype='step', linewidth=4) #, density=True)
    
    axs3.hist(real_dict_50[to_plot_r[0]][real_dict_50[to_plot_r[1]]< thr_ecal], 
              bins=bbins, range=range_, color='darkorange', alpha=0.2) #, density=True)
    axs3.hist(fake_dict_50[to_plot[0]][fake_dict_50[to_plot[1]]< thr_ecal], 
              bins=bbins, range=range_, color='darkorange', label='50 GeV', histtype='step', linewidth=4) #, density=True)
    
    axs3.hist(real_dict_85[to_plot_r[0]][real_dict_85[to_plot_r[1]]< thr_ecal], 
              bins=bbins, range=range_, color='purple', alpha=0.2) #, density=True)
    axs3.hist(fake_dict_85[to_plot[0]][fake_dict_85[to_plot[1]]< thr_ecal], 
              bins=bbins, range=range_, color='purple', label='85 GeV', histtype='step', linewidth=4) #, density=True) 
    
    # p.legend(fontsize=font)
    axs3.set_title(f"Visible energy ECal < {thr_ecal:.0f} [MeV]", fontsize=font) 
    axs3.set_xlabel("Visible energy HCal [MeV]", fontsize=font) 
    axs3.set_ylabel("Counts", fontsize=font) 
    plt.legend()
    plt.tight_layout()   
    plt.savefig(f"{my_dir}/Corr_multiEn.pdf")
    plt.close()   

b=29
bins_y = 78
plt.clf()
directory = '/data/dust/user/mmozzani/pion-clouds/figs/occ-scale/'+name_folder+'/'
os.makedirs(directory+'Features_Comparison/', exist_ok=True)
directory_2 = directory+'Features_Comparison/'
"""
print('    Resolution and Linearity ALL... ')
plotting_energy_resolution_linearity_all(my_dir = directory_2)
plt.clf()

print('    Resolution and Linearity... ')
plotting_energy_resolution_linearity(my_dir = directory_2)
plt.clf()

print('     Energy Sum... ')
plotting_energy_sum(my_dir = directory_2)
plt.clf()

print('     Hits... ')
plotting_hits(my_dir = directory_2)
plt.clf()
"""
print('     Corr... ')
plotting_correlations(my_dir = directory_2)
plt.clf()

