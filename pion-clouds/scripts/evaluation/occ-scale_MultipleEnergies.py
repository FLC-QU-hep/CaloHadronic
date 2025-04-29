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
font = 25
Ymin, Ymin_hcal, Ymax = 0, 30, 78
#save arrays 
save_dir_fake_15 = '../files/projected_array/'+name_folder+'/15GeV_'+str(shw)
save_dir_fake_50 = '../files/projected_array/'+name_folder+'/50GeV_'+str(shw) 
save_dir_fake_85 = '../files/projected_array/'+name_folder+'/85GeV_'+str(shw)  

save_dir_real_15 = '../files/projected_array/Geant4/15GeV_'+str(shw)  
save_dir_real_50 = '../files/projected_array/Geant4/50GeV_'+str(shw)  
save_dir_real_85 = '../files/projected_array/Geant4/85GeV_'+str(shw)   
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

def plotting_energy_sum(my_dir):
    fig = plt.figure(10, figsize=(25,9))
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
        pgen = axs[0].hist([0,0], bins=1, label='gen', histtype='step', color='gray') 
        axs[0].set_xlim(_range[0], _range[1])         
        axs[0].set_xlabel('Energy Sum '+title_list[j], fontsize=font)
        axs[0].set_ylabel('$\#$ Showers', fontsize=font)
        axs[0].tick_params(axis="x", labelsize=font) 
        axs[0].tick_params(axis="y", labelsize=font)  
        
        h, l = axs[0].get_legend_handles_labels()
        # ax2 = axs[0].twinx()
        if j==0: axs[0].legend([h[3],h[4]],[l[3],l[4]], loc=1, ncols=2, fontsize=font) 
        else: axs[0].legend([h[0],h[1],h[2]],[l[0],l[1],l[2]], loc=1, fontsize=font)
        # ax2.set_axis_off()
        del h, l
    plt.tight_layout()   
    plt.savefig(my_dir+'/EnergySum_'+str(shw)+ '_showers.pdf', dpi=100, bbox_inches='tight')
    plt.close()
    
def plotting_hits(my_dir):
    plt.figure(2, figsize=(25,9))
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
        axs[0].hist([0,0], bins=1, label='gen', histtype='step', color='gray')  
        axs[0].set_xlim(_range[0], _range[1])         
        axs[0].set_xlabel('# Hits '+title_list[j], fontsize=font)
        axs[0].set_ylabel('$\#$ Showers', fontsize=font)
        axs[0].tick_params(axis="x", labelsize=font)
        axs[0].tick_params(axis="y", labelsize=font)
        
        h, l = axs[0].get_legend_handles_labels()
        # ax2 = axs[0].twinx()
        if j==0: axs[0].legend([h[3],h[4]],[l[3],l[4]], loc=1, ncols=2, fontsize=font) 
        else: axs[0].legend([h[0],h[1],h[2]],[l[0],l[1],l[2]], loc=1, fontsize=font)
        # ax2.set_axis_off()
    plt.tight_layout()   
    plt.savefig(my_dir+'/Hits_'+str(shw)+ '_showers.pdf', dpi=100, bbox_inches='tight')     
    plt.close()   

def plotting_correlations(my_dir): 
    _range= [0, 1900] 
    thebins = [50, 50]
    cmin = 3
    to_plot_r = ["e_sum_hcal_list_r","e_sum_ecal_list_r"]
    to_plot = ["e_sum_hcal_list","e_sum_ecal_list"]
    
    fig = plt.figure(10, figsize=(25,9))
    gs = gridspec.GridSpec(1, 3)
    H_15, _, _ = np.histogram2d(fake_dict_15[to_plot[0]], fake_dict_15[to_plot[1]], bins=thebins, range=[_range, _range])
    H_r_15, _, _ = np.histogram2d(real_dict_15[to_plot_r[0]], real_dict_15[to_plot_r[1]], bins=thebins, range=[_range, _range]) 
    H_50, _, _ = np.histogram2d(fake_dict_50[to_plot[0]], fake_dict_50[to_plot[1]], bins=thebins, range=[_range, _range])
    H_r_50, _, _ = np.histogram2d(real_dict_50[to_plot_r[0]], real_dict_50[to_plot_r[1]], bins=thebins, range=[_range, _range])
    H_85, _, _ = np.histogram2d(fake_dict_85[to_plot[0]], fake_dict_85[to_plot[1]], bins=thebins, range=[_range, _range])
    H_r_85, _, _ = np.histogram2d(real_dict_85[to_plot_r[0]], real_dict_85[to_plot_r[1]], bins=thebins, range=[_range, _range])
    cmax = max(H_15.max(), H_r_15.max(), H_50.max(), H_r_50.max(), H_85.max(), H_r_85.max())
    
    axs1 = plt.subplot(gs[0])
    hist1 = axs1.hist2d(
        real_dict_15[to_plot_r[0]], real_dict_15[to_plot_r[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, cmap='Greens', norm=mpl.colors.LogNorm(),
    )
    axs1.hist2d(
        real_dict_50[to_plot_r[0]], real_dict_50[to_plot_r[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=mpl.colors.LogNorm(), cmap='Oranges',
    )
    axs1.hist2d(
        real_dict_85[to_plot_r[0]], real_dict_85[to_plot_r[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=mpl.colors.LogNorm(), cmap='BuPu_r',
    )
    
    axs1.set_title("Geant4", fontsize=font)
    axs1.set_xlabel("energy hcal [MeV]", fontsize=font)
    axs1.set_ylabel("energy ecal [MeV]", fontsize=font)
    # fig.colorbar(hist1[3], ax=axs1, location='bottom')
    
    axs2 = plt.subplot(gs[1])
    axs2.hist2d(
        fake_dict_15[to_plot[0]], fake_dict_15[to_plot[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=mpl.colors.LogNorm(), cmap='Greens',
    )
    hist2 =axs2.hist2d(
        fake_dict_50[to_plot[0]], fake_dict_50[to_plot[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=mpl.colors.LogNorm(), cmap='Oranges',
    )
    axs2.hist2d(
        fake_dict_85[to_plot[0]], fake_dict_85[to_plot[1]], bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=mpl.colors.LogNorm(), cmap='BuPu_r',
    )
    
    axs2.set_title("CaloHadronic", fontsize=font)
    axs2.set_xlabel("energy hcal [MeV]", fontsize=font)
    axs2.set_ylabel("energy ecal [MeV]", fontsize=font)
    # fig.colorbar(hist2[3], ax=axs2, location='bottom') 
    
    axs3 = plt.subplot(gs[2])    

    thr_ecal = 30
    range_ = [0, 2000]
    bbins= 40
    axs3.hist(real_dict_15[to_plot_r[0]][real_dict_15[to_plot_r[1]]> thr_ecal], 
              bins=bbins, range=range_, color='green', alpha=0.2, density=True)
    axs3.hist(fake_dict_15[to_plot[0]][fake_dict_15[to_plot[1]]> thr_ecal], 
              bins=bbins, range=range_, color='green', label='15 GeV', histtype='step', linewidth=4, density=True)
    
    axs3.hist(real_dict_50[to_plot_r[0]][real_dict_50[to_plot_r[1]]> thr_ecal], 
              bins=bbins, range=range_, color='orange', alpha=0.3, density=True)
    axs3.hist(fake_dict_50[to_plot[0]][fake_dict_50[to_plot[1]]> thr_ecal], 
              bins=bbins, range=range_, color='orange', label='50 GeV', histtype='step', linewidth=4, density=True)
    
    axs3.hist(real_dict_85[to_plot_r[0]][real_dict_85[to_plot_r[1]]> thr_ecal], 
              bins=bbins, range=range_, color='purple', alpha=0.2, density=True)
    axs3.hist(fake_dict_85[to_plot[0]][fake_dict_85[to_plot[1]]> thr_ecal], 
              bins=bbins, range=range_, color='purple', label='85 GeV', histtype='step', linewidth=4, density=True) 
    
    # p.legend(fontsize=font)
    axs3.set_title(f"energy ecal < {thr_ecal:.0f} [MeV]", fontsize=font) 
    axs3.set_xlabel("energy hcal [MeV]", fontsize=font) 
    axs3.set_ylabel("Normalized", fontsize=font) 
    plt.legend()
    plt.savefig(f"{my_dir}/Corr_multiEn.pdf")
    plt.close()   

b=29
bins_y = 78
plt.clf()
directory = '../figs/occ-scale/'+name_folder+'/'
os.makedirs(directory+'Features_Comparison/', exist_ok=True)
directory_2 = directory+'Features_Comparison/'

print('     Energy Sum...')
plotting_energy_sum(my_dir = directory_2)
plt.clf()
print('     Hits...')
plotting_hits(my_dir = directory_2)
plt.clf()
print('     Corr...')
plotting_correlations(my_dir = directory_2)
plt.clf()

