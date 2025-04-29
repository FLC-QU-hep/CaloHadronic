
from tqdm import tqdm
import numpy as np
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import binned_statistic, pearsonr
import utils.metrics as metrics
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from pathlib import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.gridspec as gridspec
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import pandas as pd
import json

mpl.rcParams['xtick.labelsize'] = 18    
mpl.rcParams['ytick.labelsize'] = 18
# mpl.rcParams['font.size'] = 28
mpl.rcParams['font.size'] = 20
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['axes.aspect'] = 'equal' 
# mpl.rcParams['figure.autolayout'] = True # for the hit e distr.
# mpl.rcParams['font.weight'] = 1  # Or you can use a numerical value, e.g., 700
plt.close() 

class Configs(): 
    def __init__(self):
    # legend font
        self.font = font_manager.FontProperties(
            family='serif',
            size=25
        )
        self.text_size = 25
        
        self.title = ' ' 
        # self.title = '$\textbf{full spectrum}$'
        # self.title = r'\textbf{10 GeV @ (0,-72)}'
        self.font_ratio = 25
        self.font_labely = 25
        self.font_labelx = 25
        # error band
        self.alpha = 0.3
        self.lw = 5

    # radial profile
        self.bins_r = np.linspace(0, 25, 26) #np.logspace(np.log10(0.1), np.log10(40), 60)
        self.origin = (0, -72) # (0, 13) for 50 GeV
        self.xrange_radial = [0, 30]
        self.ylim_radial = (2e-1, 2e3)
        # self.origin = (0, 0)
        # self.origin = (3.754597092*10, -3.611833191*10)
        
    # occupancy
        # self.occup_bins = np.linspace(150, 1419, 100) # for 50 GeV @ (0,13)
        # self.occup_bins = np.linspace(-200, 250, 100) # for 1 GeV @ (0,13)
        # self.occup_bins = np.linspace(-100, 2200, 300) # for full spectrum 
        self.occup_bins = np.linspace(-100, 1000, 50) # for 30 GeV @ (0,-50)
        self.plot_text_occupancy = False
        self.occ_indent = 20
    # Nhits 
        self.occ_bins = np.linspace(-2, 3000, 40)
        self.ylim_Nhits = (0, 3000) 
    # e_sum
        # self.e_sum_bins = np.linspace(20.01, 2200, 100)
        self.e_sum_bins = np.linspace(0, 2300, 25) # for full spectrum
        # self.e_sum_bins = np.linspace(-100, 1000, 50) # for 30 GeV @ (0,-50)
        # self.e_sum_bins = np.linspace(600, 1700, 100) # for 50 GeV @ (0,13)
        # self.e_sum_bins = np.linspace(-200, 250, 100) # for 1 GeV @ (0,13)
        self.ylim_esum = (20, 3e3)
    # e ratio
        self.ratio_bins = np.logspace(np.log10(0.002), np.log10(0.09), 40)
    # start L
        self.startL_bins = 36 #78 # for full spectrum
        self.plot_text_e = False
        self.plot_legend_e = True
        self.e_indent = 20

    # hits
        self.hit_bins = np.logspace(np.log10(0.01000001), np.log10(200), 60)
        # self.hit_bins = np.logspace(np.log10(0.01), np.log10(100), 70)
        self.ylim_hits = (100, 1e7)
        # self.ylim_hits = (10, 8*1e5)
    #x y z
        self.bins_xz = 35
        self.bins_y = 78
        self.xyz_ranges = [(0, 180), (0, 77), (0, 180)] 
        
    #CoG
        self.bins_cog = 30  
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        # self.cog_ranges = [(-150, 150), (1800, 1950), (-200, 150)] # full spectrum angular conditioned
        # self.cog_ranges = [(0, 50), (1800, 1950), (-30, 100)]
        self.cog_ranges = [(-8, 8), (9, 76), (-8, 8)]
        # self.cog_ranges = [(-1.7, 1.2), (1891, 1949), (38.5, 41.99)]
        # self.cog_ranges = [(-3.99, 3.99), (1861, 1999), (36.01, 43.99)]
        # self.cog_ranges = [(33.99, 39.99), (1861, 1999), (-38.9, -32.9)]

    # xyz featas
        self.bins_feats = 50  
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        # self.feats_ranges = [(-200, 200), (1811, 2011), (-250, 300)]
        self.feats_ranges = [(-250, 250), (1800, 2011), (-250, 250)]
        # self.cog_ranges = [(-3.99, 3.99), (1861, 1999), (36.01, 43.99)]
        # self.cog_ranges = [(33.99, 39.99), (1861, 1999), (-38.9, -32.9)]
    # all
        self.threshold = 0.1   # MeV / half a MIP
        #self.color_lines = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        self.color_lines = ['tab:green', 'tab:orange', 'tab:blue', 'tab:red', 'tab:purple', 'tab:pink']
        # self.color_lines = ['tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']
        self.include_artifacts = False   # include_artifacts = True -> keeps points that hit dead material
    # percentile edges for occupancy based metrics
        self.layer_edges = [0, 8, 11, 13, 15, 16, 18, 19, 21, 24, 29]
        self.radial_edges = [0, 6.558, 9.849, 12.96, 17.028, 23.434, 33.609, 40.119, 48.491, 68.808, 300]

plt_config = Configs() 

def ratio_plots(axs, geant4_data, gen_data_list, err_data, err_gen_list, bins, pos, plt_config=plt_config):
    # ratio plot on the bottom
    lims_min = 0.6
    lims_max = 1.4
    eps = 1e-5
    
    for i, gen_data in enumerate(gen_data_list):
        centers = pos 
        ratios = np.clip((gen_data+eps)/(geant4_data+eps), lims_min, lims_max) 
        mask = (ratios > lims_min) & (ratios < lims_max)  # mask ratios within plotting y range
        # only connect dots with adjecent points 
        starts = np.argwhere(np.insert(mask[:-1],0,False)<mask)[:,0] 
        ends = np.argwhere(np.append(mask[1:],False)<mask)[:,0]+1 
        indexes = np.stack((starts,ends)).T 
        for idxs in indexes: 
            sub_mask = np.zeros(len(mask), dtype=bool) 
            sub_mask[idxs[0]:idxs[1]] = True  
            # axs.stairs(ratios[sub_mask], edges=bins, color=plt_config.color_lines[i], lw=2)
            axs.stairs(ratios, edges=bins, color=plt_config.color_lines[i], lw=2)  
        # remaining points either above or below plotting y range
        mask = (ratios == lims_min)
        axs.plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='v', color=plt_config.color_lines[i], clip_on=False)
        mask = (ratios == lims_max) 
        axs.plot(centers[mask], ratios[mask], linestyle='', lw=2, marker='^', color=plt_config.color_lines[i], clip_on=False)

        # plot error band 
        ratio_err = ratios * np.sqrt(
            ((err_gen_list[i]) / (gen_data)) ** 2 + ((err_data) / (geant4_data)) ** 2
        )
        axs.stairs(
            ratios + ratio_err,
            edges=bins,
            baseline=ratios - ratio_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

    axs.set_ylim(lims_min, lims_max)
    # horizontal line at 1
    axs.axhline(1, linestyle='-', lw=1, color='k')

def plt_cog(cog, cog_list, shw, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    lables = ["X", "Y", "Z"] # local coordinate system  
    # plt.figure(figsize=(21, 9)) 
    fig, axs = plt.subplots(2, 3, figsize=(25, 10), sharex='col', height_ratios=[3, 1])

    for k, j in enumerate([0, 2, 1]):
        axs[0, k].set_xlim(plt_config.cog_ranges[j])
        # real data
        h_data = axs[0, k].hist(
            np.array(cog[j]),
            bins=plt_config.bins_cog,
            color="lightgrey",
            range=plt_config.cog_ranges[j],
            rasterized=True,
        )
        # uncertainty band
        err_data = np.sqrt(h_data[0])
        axs[0, k].stairs(
            h_data[0] + err_data,
            edges=h_data[1],
            baseline=h_data[0] - err_data,
            color="dimgrey",
            lw=2,
            hatch="///",
        )

        # generated data
        data_gen_list, err_gen_list = [] , []
        for i, cog_ in enumerate(cog_list[j]):
            h_gen = axs[0, k].hist(
                np.array(cog_),
                bins=h_data[1],
                histtype="step",
                # linestyle="-",
                linewidth=3,
                color=plt_config.color_lines[i],
                range=plt_config.cog_ranges[j],
            )
            data_gen_list.append(h_gen[0]) 
            # uncertainty band in histogram
            h_gen_err = np.sqrt(h_gen[0])
            axs[0, k].stairs(
                h_gen[0] + h_gen_err,
                edges=h_gen[1],
                baseline=h_gen[0] - h_gen_err,
                color=plt_config.color_lines[i],
                alpha=plt_config.alpha,
                fill=True,
            )
            err_gen_list.append(h_gen_err[0])
            
            # for legend ##############################################
            if j == 2:
            #     plt.plot(0, 0, lw=2, color='black', label=labels[0])
                axs[0, k].hist(np.zeros(1), label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
                axs[0, k].plot(0, 0, linestyle='-', lw=3, color=plt_config.color_lines[i], label=labels[i+1])
            ###########################################################
                    
        # horizontal line at 1
        axs[1, k].axhline(1, linestyle='-', lw=1, color='k')

        # for legend ##############################################
        # plt.legend(prop=plt_config.font, loc=(0.37, 0.76))
        axs[0, 1].legend(prop=plt_config.font, loc='best')
        axs[0, k].set_title(title, fontsize=plt_config.font.get_size(), loc='right')

        ###########################################################
        ratio_plots(axs[1,k], h_data[0], data_gen_list, err_data[0], err_gen_list, h_data[1], pos=np.array((h_data[1][:-1] + h_data[1][1:])/2), plt_config=plt_config)
        axs[0, k].set_ylim(0, max(h_data[0]) + max(h_data[0])*0.5)
        #axs[1, k].set_yscale('log')
        axs[1, k].set_xlabel(f'center of gravity {lables[j]}', fontsize=plt_config.font_labelx)
        axs[0, 0].set_ylabel('# showers', fontsize=plt_config.font_labely)
        axs[1, 0].set_ylabel('ratio to G4', fontsize=plt_config.font_ratio)
    if my_dir is not None:
        # fig.tight_layout() 
        plt.savefig(my_dir+'c_cog.pdf', dpi=100, bbox_inches='tight')
    plt.close()
      
def plt_spinal(e_layers, e_layers_list, shw, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    
    fig, axs = plt.subplots(2, 1, figsize=(7,9), height_ratios=[3, 1], sharex=True)
    ## for legend ##########################################
    axs[0].hist(np.zeros(1)-10, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    for i in range(len(e_layers_list)):
        axs[0].plot(0, 0, linestyle='-', lw=3, color=plt_config.color_lines[i], label=labels[i+1])
    ########################################################

    pos = np.arange(1, len(e_layers)+1)
    bins = np.arange(0.5, len(e_layers)+1.5)

    axs[0].stairs(e_layers, edges=bins, color="lightgrey", fill=True)
    
    # uncertainty band
    err_data = e_layers.std(axis=0) / np.sqrt(shw) # std of individual measures --> std of mean
    axs[0].stairs(
        e_layers + err_data,
        edges=bins,
        baseline=e_layers - err_data,
        color="dimgrey",
        lw=2,
        hatch="///",
    )
    err_gen_list = [] 
    for i, e_layers_ in enumerate(e_layers_list):
        err_gen = e_layers_.std(axis=0) / np.sqrt(shw)
        axs[0].hist(pos, 
                    bins=bins, 
                    weights=e_layers_, 
                    histtype='step', 
                    # linestyle='-', 
                    linewidth=3, 
                    color=plt_config.color_lines[i]
        )
        axs[0].stairs(
            e_layers_ + err_gen,
            edges=bins,
            baseline=e_layers_ - err_gen,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            lw=plt_config.lw,
            fill=True,
        )
        err_gen_list.append(err_gen)
    # ratio plot on the bottom
    ratio_plots(axs[1], e_layers, e_layers_list, err_data, err_gen_list, bins, pos, plt_config=plt_config)    
    axs[0].set_yscale('log')
    axs[0].set_ylim(1.1e-1, 2e2)
    axs[0].set_xlim(0, 79)
    plt.xlabel('layers', fontsize=plt_config.font_labelx)
    axs[0].set_ylabel('mean energy [MeV]', fontsize=plt_config.font_labely)
    
    # plt.legend(prop=plt_config.font, loc=(0.35, 0.78))
    plt.legend(prop=plt_config.font, loc='best')
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc='right')
    axs[1].set_ylabel('ratio to G4', fontsize=plt_config.font_ratio)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    if my_dir is not None:
        fig.tight_layout() 
        plt.savefig(my_dir+'spinal.pdf', dpi=100, bbox_inches='tight')
    plt.close()
    
def plt_radial(e_radial, e_radial_list, events, labels, my_dir=None, cfg=plt_config, title=plt_config.title):
    fig, axs = plt.subplots(2, 1, figsize=(7,9), height_ratios=[3, 1], sharex=True)

    ## for legend ########################################## 
    axs[0].hist(np.zeros(1)-10, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2, range=(0,1))
    for i in range(len(e_radial_list)):
        axs[0].plot(0, 0, linestyle='-', lw=3, color=cfg.color_lines[i], label=labels[i+1])
    axs[0].set_title(title, fontsize=cfg.font.get_size(), loc='right')
    # plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    axs[0].legend(prop=cfg.font, loc='upper right')
    ########################################################

    # mean and std as binned statistic
    mean, edges, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r, statistic="mean"
    )
    std, _, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r, statistic="std"
    )
    count, _, _ = binned_statistic(
        e_radial[0], e_radial[1], bins=plt_config.bins_r, statistic="count"
    )

    mean_shower = mean * count / events  # mean shower energy per bin
    std_shower = std / np.sqrt(count)  # std of individual event measures --> std of mean
    std_shower = std_shower * count / events  # std of shower energy per bin

    # data histogram
    axs[0].stairs(mean_shower, edges=edges, color="lightgrey", fill=True)
    # uncertainty band
    axs[0].stairs(
        mean_shower + std_shower,
        edges=edges,
        baseline=mean_shower - std_shower,
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    mean_shower_gen_list, err_gen_list = [], []
    for i, e_radial_ in enumerate(e_radial_list):
        # mean and std as binned statistic
        mean, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="mean"
        )
        std, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="std"
        )
        count, _, _ = binned_statistic(
            e_radial_[0], e_radial_[1], bins=edges, statistic="count"
        )
        mean_shower_gen = mean * count / events  # mean shower energy per bin
        mean_shower_gen_list.append(mean_shower_gen)
        std_shower_gen = std / np.sqrt(count)  # std of individual event measures --> std of mean
        std_shower_gen = std_shower_gen * count / events  # std of shower energy per bin
        err_gen_list.append(std_shower_gen)
        # gen histogram
        axs[0].stairs(
            mean_shower_gen,
            edges=edges,
            linestyle="-",
            lw=3,
            color=plt_config.color_lines[i],
        )
        # uncertainty band
        axs[0].stairs(
            mean_shower_gen + std_shower_gen,
            edges=edges,
            baseline=mean_shower_gen - std_shower_gen,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )

    # ratio plot on the bottom
    ratio_plots(axs[1], mean_shower, mean_shower_gen_list, std_shower, err_gen_list, edges, pos=np.array((edges[:-1] + edges[1:])/2), plt_config=plt_config)  
        
    axs[0].set_ylim(plt_config.ylim_radial[0], plt_config.ylim_radial[1])
    axs[0].set_yscale('log')
    # axs[0].set_xscale("log")
    plt.xlabel("radius [mm]", fontsize=plt_config.font_labelx)
    axs[0].set_ylabel('mean energy [MeV]', fontsize=plt_config.font_labely)
    axs[1].set_ylabel('ratio to G4', fontsize=plt_config.font_ratio)
    plt.subplots_adjust(hspace=0.1)
    
    if my_dir is not None:
        fig.tight_layout() 
        plt.savefig(my_dir+'radial.pdf', dpi=100, bbox_inches='tight')
    plt.close()
    
def plt_hit_e(hits, hits_list, events, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    # for legend ##########################################
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=3
    )
    for i in range(len(hits_list)):
        axs[0].plot( 
            0, 
            0, 
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    plt.legend(prop=plt_config.font, loc="upper right")
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    ########################################################

    h_data = axs[0].hist(
        hits, bins=plt_config.hit_bins, color="lightgrey", rasterized=True
    )

    # uncertainty band
    h_data_err = np.sqrt(h_data[0])
    axs[0].stairs(
        h_data[0] + h_data_err,
        edges=h_data[1],
        baseline=h_data[0] - h_data_err,
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    
    data_gen_list, err_gen_list = [], []
    for i, hits_ in enumerate(hits_list):
        h_gen = axs[0].hist(
            hits_,
            bins=h_data[1],
            histtype="step",
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
        )
        data_gen_list.append(h_gen[0])

        # uncertainty band in histogram
        h_gen_err = np.sqrt(h_gen[0])
        axs[0].stairs(
            h_gen[0] + h_gen_err,
            edges=h_gen[1],
            baseline=h_gen[0] - h_gen_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )
        err_gen_list.append(h_gen_err)

    # ratio plot on the bottom
    ratio_plots(axs[1], h_data[0], data_gen_list, h_data_err, err_gen_list, h_data[1], pos=np.array((h_data[1][:-1] + h_data[1][1:])/2), plt_config=plt_config)  
     
    # axs[0].axvspan(
    #     h_data[1].min(),
    #     plt_config.threshold,
    #     facecolor="gray",
    #     alpha=0.5,
    #     hatch="/",
    #     edgecolor="k",
    # )
    # axs[1].axvspan(
    #     h_data[1].min(),
    #     plt_config.threshold,
    #     facecolor="gray",
    #     alpha=0.5,
    #     hatch="/",
    #     edgecolor="k",
    # )
    axs[0].set_xlim(h_data[1].min(), h_data[1].max())
    # axs[0].set_xlim(h_data[1].min(), 3e2)
    axs[0].set_ylim(plt_config.ylim_hits[0], plt_config.ylim_hits[1])
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_ylabel("# cells", fontsize=plt_config.font_labelx)
    axs[1].set_ylabel("ratio to G4", fontsize=plt_config.font_labelx)
    plt.xlabel("visible cell energy [MeV]", fontsize=plt_config.font_labelx)

    if my_dir is not None:
        fig.tight_layout() 
        plt.savefig(my_dir+'hits.pdf', dpi=100, bbox_inches='tight')
    plt.close()

def plt_sum_e(e_sum, e_sum_list, events, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    # for legend ##########################################
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=3
    )
    for i in range(len(e_sum_list)):
        axs[0].plot( 
            0, 
            0, 
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    plt.legend(prop=plt_config.font, loc="upper right")
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    ########################################################
    # _range= [0, 2.3e3]
    h_data = axs[0].hist(
        e_sum, bins=plt_config.e_sum_bins, color="lightgrey", rasterized=True
    )

    # uncertainty band
    h_data_err = np.sqrt(h_data[0])
    axs[0].stairs(
        h_data[0] + h_data_err,
        edges=h_data[1],
        baseline=h_data[0] - h_data_err,
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    
    data_gen_list, err_gen_list = [], []
    for i, e_sum_ in enumerate(e_sum_list):
        h_gen = axs[0].hist(
            e_sum_,
            bins=h_data[1],
            histtype="step",
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
        )
        data_gen_list.append(h_gen[0])

        # uncertainty band in histogram
        h_gen_err = np.sqrt(h_gen[0])
        axs[0].stairs(
            h_gen[0] + h_gen_err,
            edges=h_gen[1],
            baseline=h_gen[0] - h_gen_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )
        err_gen_list.append(h_gen_err)

    # ratio plot on the bottom
    ratio_plots(axs[1], h_data[0], data_gen_list, h_data_err, err_gen_list, h_data[1], pos=np.array((h_data[1][:-1] + h_data[1][1:])/2), plt_config=plt_config)  

    # axs[0].set_ylim(plt_config.ylim_hits[0], plt_config.ylim_hits[1])
    # axs[0].set_yscale("log")
    axs[0].set_ylabel("# showers", fontsize=plt_config.font_labelx)
    axs[1].set_ylabel("ratio to G4", fontsize=plt_config.font_labelx)
    plt.xlabel("visible energy [MeV]", fontsize=plt_config.font_labelx)

    if my_dir is not None:
        # fig.tight_layout() 
        plt.savefig(my_dir+'e_sum.pdf', dpi=100, bbox_inches='tight')
    plt.close()

def plt_e_ratio(e_r, e_r_list, events, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    # for legend ##########################################
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=3
    )
    for i in range(len(e_r_list)):
        axs[0].plot( 
            0, 
            0, 
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    plt.legend(prop=plt_config.font, loc="upper right")
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    ########################################################

    h_data = axs[0].hist(
        e_r, bins=plt_config.ratio_bins , color="lightgrey", rasterized=True
    )

    # uncertainty band
    h_data_err = np.sqrt(h_data[0])
    axs[0].stairs(
        h_data[0] + h_data_err,
        edges=h_data[1],
        baseline=h_data[0] - h_data_err,
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    
    data_gen_list, err_gen_list = [], []
    for i, e_r_ in enumerate(e_r_list):
        h_gen = axs[0].hist(
            e_r_,
            bins=h_data[1],
            histtype="step",
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
        )
        data_gen_list.append(h_gen[0])

        # uncertainty band in histogram
        h_gen_err = np.sqrt(h_gen[0])
        axs[0].stairs(
            h_gen[0] + h_gen_err,
            edges=h_gen[1],
            baseline=h_gen[0] - h_gen_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )
        err_gen_list.append(h_gen_err)

    # ratio plot on the bottom
    ratio_plots(axs[1], h_data[0], data_gen_list, h_data_err, err_gen_list, h_data[1], pos=np.array((h_data[1][:-1] + h_data[1][1:])/2), plt_config=plt_config)  

    # axs[0].set_xlim(h_data[1].min(), 3e2)
    # axs[0].set_ylim(plt_config.ylim_hits[0], plt_config.ylim_hits[1])
    # axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_xlim(plt_config.ratio_bins[0], plt_config.ratio_bins[-1])
    axs[0].set_ylabel("# showers", fontsize=plt_config.font_labelx)
    axs[1].set_ylabel("ratio to G4", fontsize=plt_config.font_labelx)
    plt.xlabel("Energy Ratio", fontsize=plt_config.font_labelx)

    if my_dir is not None:
        fig.tight_layout() 
        plt.savefig(my_dir+'energy_ratio.pdf', dpi=100, bbox_inches='tight')
    plt.close()
    
def plt_Nhits(occ, occ_list, events, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    # for legend ##########################################
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=3
    )
    for i in range(len(occ_list)):
        axs[0].plot( 
            0, 
            0, 
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    plt.legend(prop=plt_config.font, loc="upper right")
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    ########################################################

    h_data = axs[0].hist(
        occ, bins=plt_config.occ_bins, color="lightgrey", rasterized=True
    )

    # uncertainty band
    h_data_err = np.sqrt(h_data[0])
    axs[0].stairs(
        h_data[0] + h_data_err,
        edges=h_data[1],
        baseline=h_data[0] - h_data_err,
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    
    data_gen_list, err_gen_list = [], []
    for i, occ_ in enumerate(occ_list):
        h_gen = axs[0].hist(
            occ_,
            bins=h_data[1],
            histtype="step",
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
        )
        data_gen_list.append(h_gen[0])

        # uncertainty band in histogram
        h_gen_err = np.sqrt(h_gen[0])
        axs[0].stairs(
            h_gen[0] + h_gen_err,
            edges=h_gen[1],
            baseline=h_gen[0] - h_gen_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )
        err_gen_list.append(h_gen_err)

    # ratio plot on the bottom
    ratio_plots(axs[1], h_data[0], data_gen_list, h_data_err, err_gen_list, h_data[1], pos=np.array((h_data[1][:-1] + h_data[1][1:])/2), plt_config=plt_config)  

    # axs[0].set_xlim(h_data[1].min(), h_data[1].max())
    # axs[0].set_xlim(h_data[1].min(), 3e2)
    axs[0].set_ylim(plt_config.ylim_Nhits[0], plt_config.ylim_Nhits[1])
    # axs[0].set_yscale("log")
    axs[0].set_ylabel("# showers", fontsize=plt_config.font_labelx)
    axs[1].set_ylabel("ratio to G4", fontsize=plt_config.font_labelx)
    plt.xlabel("number of hits", fontsize=plt_config.font_labelx)

    if my_dir is not None:
        # fig.tight_layout() 
        plt.savefig(my_dir+'Nhits.pdf', dpi=100, bbox_inches='tight')
    plt.close()

def plt_startlayer(startL, startL_list, events, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    fig, axs = plt.subplots(2, 1, figsize=(7, 9), height_ratios=[3, 1], sharex=True)
    # for legend ##########################################
    axs[0].hist(
        np.zeros(1) + 1, label=labels[0], color="lightgrey", edgecolor="dimgrey", lw=3
    )
    for i in range(len(startL_list)):
        axs[0].plot( 
            0, 
            0, 
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
            label=labels[i + 1],
        )
    plt.legend(prop=plt_config.font, loc="upper right")
    axs[0].set_title(title, fontsize=plt_config.font.get_size(), loc="right")
    ########################################################
    print("startL", startL.shape)
    print(startL.min(), startL.max())
    print(startL[:20])
    print("startL_list", startL_list[0].shape)
    h_data = axs[0].hist(
        startL, bins=plt_config.startL_bins, color="lightgrey", rasterized=True
    )
    print("h_data 0", h_data[0].shape)
    print("h_data 1", h_data[1].shape)
    # uncertainty band
    h_data_err = np.sqrt(h_data[0])
    axs[0].stairs(
        h_data[0] + h_data_err,
        edges=h_data[1],
        baseline=h_data[0] - h_data_err,
        color="dimgrey",
        lw=3,
        hatch="///",
    )
    
    data_gen_list, err_gen_list = [], []
    for i, startL_ in enumerate(startL_list):
        h_gen = axs[0].hist(
            startL_,
            bins=h_data[1],
            histtype="step",
            # linestyle="-",
            linewidth=3,
            color=plt_config.color_lines[i],
        )
        data_gen_list.append(h_gen[0])

        # uncertainty band in histogram
        h_gen_err = np.sqrt(h_gen[0])
        axs[0].stairs(
            h_gen[0] + h_gen_err,
            edges=h_gen[1],
            baseline=h_gen[0] - h_gen_err,
            color=plt_config.color_lines[i],
            alpha=plt_config.alpha,
            fill=True,
        )
        err_gen_list.append(h_gen_err)

    # ratio plot on the bottom
    ratio_plots(axs[1], h_data[0], data_gen_list, h_data_err, err_gen_list, h_data[1], pos=np.array((h_data[1][:-1] + h_data[1][1:])/2), plt_config=plt_config)  

    # axs[0].set_xlim(h_data[1].min(), h_data[1].max())
    # axs[0].set_xlim(h_data[1].min(), 3e2)
    # axs[0].set_ylim(plt_config.ylim_hits[0], plt_config.ylim_hits[1])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("# showers", fontsize=plt_config.font_labelx)
    axs[1].set_ylabel("ratio to G4", fontsize=plt_config.font_labelx)
    plt.xlabel("shower start layer [layer]", fontsize=plt_config.font_labelx)

    if my_dir is not None:
        # fig.tight_layout() 
        plt.savefig(my_dir+'startL.pdf', dpi=100, bbox_inches='tight')
    plt.close()  
 
def plt_xyz(xyz, xyz_list, shw, labels, my_dir=None, plt_config=plt_config, title=plt_config.title):
    lables = ["X", "Y", "Z"] # local coordinate system  
    # plt.figure(figsize=(21, 9)) 
    fig, axs = plt.subplots(2, 3, figsize=(25, 10), sharex='col', height_ratios=[3, 1])

    for k, j in enumerate([0, 2, 1]):
        # axs[0, k].set_xlim(plt_config.cog_ranges[j])
        if j==1: bbb = plt_config.bins_y
        else: bbb = plt_config.bins_xz
        # real data
        h_data = axs[0, k].hist(
            np.array(xyz[j]),
            bins=bbb,
            color="lightgrey",
            range=plt_config.xyz_ranges[j],
            rasterized=True,
        )
        # uncertainty band
        err_data = np.sqrt(h_data[0])
        axs[0, k].stairs(
            h_data[0] + err_data,
            edges=h_data[1],
            baseline=h_data[0] - err_data,
            color="dimgrey",
            lw=2,
            hatch="///",
        )

        # generated data
        data_gen_list, err_gen_list = [] , []
        for i, xyz_ in enumerate(xyz_list[j]):
            h_gen = axs[0, k].hist(
                np.array(xyz_),
                bins=h_data[1],
                histtype="step",
                # linestyle="-",
                linewidth=3,
                color=plt_config.color_lines[i],
                range=plt_config.xyz_ranges[j],
            )
            data_gen_list.append(h_gen[0]) 
            # uncertainty band in histogram
            h_gen_err = np.sqrt(h_gen[0])
            axs[0, k].stairs(
                h_gen[0] + h_gen_err,
                edges=h_gen[1],
                baseline=h_gen[0] - h_gen_err,
                color=plt_config.color_lines[i],
                alpha=plt_config.alpha,
                fill=True,
            )
            err_gen_list.append(h_gen_err[0])
            
            # for legend ##############################################
            if j == 1:
            #     plt.plot(0, 0, lw=2, color='black', label=labels[0])
                axs[0, k].hist(np.zeros(1)-10, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
                axs[0, k].plot(0, 0, linestyle='-', lw=3, color=plt_config.color_lines[i], label=labels[i+1])
            ###########################################################
                    
        # horizontal line at 1
        axs[1, k].axhline(1, linestyle='-', lw=1, color='k')

        # for legend ##############################################
        # plt.legend(prop=plt_config.font, loc=(0.37, 0.76))
        axs[0, 2].legend(prop=plt_config.font, loc='best')
        # axs[0, k].set_title(title, fontsize=plt_config.font.get_size(), loc='right')
        axs[0, k].set_yscale('log')
        ###########################################################
        ratio_plots(axs[1,k], h_data[0], data_gen_list, err_data[0], err_gen_list, h_data[1], pos=np.array((h_data[1][:-1] + h_data[1][1:])/2), plt_config=plt_config)
        # axs[0, k].set_ylim(0, max(h_data[0]) + max(h_data[0])*0.5)
        axs[0, k].set_ylim([1e3, 4e7])
        
        if j==1: unit = '[layers]'
        else: unit= '[bins]'
        axs[1, k].set_xlabel(f'{lables[j]} '+unit, fontsize=plt_config.font_labelx)
        axs[0, 0].set_ylabel('# showers', fontsize=plt_config.font_labely)
        axs[1, 0].set_ylabel('ratio to G4', fontsize=plt_config.font_ratio)
        
    if my_dir is not None:
        fig.tight_layout() 
        plt.savefig(my_dir+'xyz.pdf', dpi=100,  bbox_inches='tight')
    plt.close()
    
def calc_wdist_2d(obs_real, obs_model, iterations=5, batch_size=100):
    means, stds = [], []
    for i in range(obs_real.shape[1]):
        j = 0
        wdists = []
        for _ in range(iterations):
            wdist = wasserstein_distance(obs_real[j:j+batch_size,i], obs_model[j:j+batch_size,i])
            wdists.append(wdist)
            j += batch_size
        # print(f'feature {i}: {np.mean(wdists)} +- {np.std(wdists)}')
        means.append(np.mean(wdists))
        stds.append(np.std(wdists))
    return np.array(means), np.array(stds)

def calc_wdist_1d(obs_real, obs_model, uw=None, vw=None, tot_shw=100, iterations = 5):
    # note: quatiles is used onlt for radial and longitudinal profiles
    # for this we define ten radial and longitudinal energy observables for the calorimeter showers
    # Respectively, the ten observables are defined such that energy is clustered in each observable
    # with an equal amount of statistics. Put differently, the energy is binned in ten quantiles with
    # approximately the same number of cell hits in each quantile
    j = 0 
    batch_size = int(tot_shw/iterations)
    np.random.shuffle(obs_real)
    np.random.shuffle(obs_model)
    wdists = [] 
    
    for _ in range(iterations):
        # r = np.zeros_like(obs_real[j:j+batch_size])
        # weights_r = np.ones_like(obs_real[j:j+batch_size])
        if uw is not None:
            if len(uw[j:j+batch_size]) == 0:
                continue  # or handle it some other way

            wdist = wasserstein_distance(obs_real[j:j+batch_size], obs_model[j:j+batch_size], u_weights= uw[j:j+batch_size], v_weights = vw[j:j+batch_size])
            # wd_ur = wasserstein_distance(obs_real[j:j+batch_size], r)
            # wd_vr = wasserstein_distance(obs_model[j:j+batch_size], r)
        else:
            wdist = wasserstein_distance(obs_real[j:j+batch_size], obs_model[j:j+batch_size])
            #    wd_ur = wasserstein_distance(obs_real[j:j+batch_size], r)
            #    wd_vr = wasserstein_distance(obs_model[j:j+batch_size], r)
        
        # normalized wasserstein distance
        # wdists.append(wdist / (wd_ur + wd_vr))
        wdists.append(wdist)
        j += batch_size
    return np.mean(wdists), np.std(wdists)

def wd_table(real_dict, fake_dict, shw, threshold, my_dir, ecal=False, hcal=False):
    my_dict={'COG x' : [],'COG y' : [],'COG z' : [], 'Shower Start Layer': [], 'Visible Cell Energy': [], 
            'Energy Sum' : [], 'N. Hits': [], 'Radial Energy': [], 'Energy along y': []}
    wd_dictionary = {}
        
    if ecal: pp,l = '_ecal', 30
    elif hcal: pp,l = '_hcal', 48
    else: pp,l = '', 78
    wd_mean_cog_x, wd_std_cog_x = calc_wdist_1d(real_dict["cog_x_r"+pp]-0.5, fake_dict["cog_x"+pp], tot_shw = shw)
    wd_mean_cog_y, wd_std_cog_y = calc_wdist_1d(real_dict["cog_y_r"+pp], fake_dict["cog_y"+pp], tot_shw = shw)
    wd_mean_cog_z, wd_std_cog_z = calc_wdist_1d(real_dict["cog_z_r"+pp]-0.5, fake_dict["cog_z"+pp], tot_shw = shw)
    dict_cog_x, dict_cog_y, dict_cog_z = '%.2f $\pm$ %.2f'%(wd_mean_cog_x, wd_std_cog_x), '%.2f $\pm$ %.2f'%(wd_mean_cog_y, wd_std_cog_y), '%.2f $\pm$ %.2f'%(wd_mean_cog_z, wd_std_cog_z)
    my_dict['COG x'].append(dict_cog_x)
    my_dict['COG y'].append(dict_cog_y)
    my_dict['COG z'].append(dict_cog_z)
    
    wd_mean_sl, wd_std_sl = calc_wdist_1d(real_dict["start_layer_list_r"+pp], fake_dict["start_layer_list"+pp], tot_shw = shw)
    dict_sl = '%.2f $\pm$ %.2f'%(wd_mean_sl, wd_std_sl)
    my_dict['Shower Start Layer'].append(dict_sl)
    
    wd_mean_hits, wd_std_hits = calc_wdist_1d(real_dict["hits"+pp+"_list_r"], fake_dict["hits"+pp+"_list"], tot_shw = shw)
    dict_hits = '%.2f $\pm$ %.2f'%(wd_mean_hits, wd_std_hits)
    my_dict['Visible Cell Energy'].append(dict_hits)
    
    wd_mean_esum, wd_std_esum = calc_wdist_1d(real_dict["e_sum"+pp+"_list_r"], fake_dict["e_sum"+pp+"_list"], tot_shw = shw)
    dict_esum = '%.2f $\pm$ %.2f'%(wd_mean_esum, wd_std_esum)
    my_dict['Energy Sum'].append(dict_esum)
    
    radial = np.linspace(0, 30, len(real_dict["e_radial"+pp+"_r"][0]))
    wd_mean_rad, wd_std_rad = calc_wdist_1d(radial, radial, uw=real_dict["e_radial"+pp+"_r"][0], vw=fake_dict["e_radial"+pp][0], tot_shw = len(real_dict["e_radial"+pp+"_r"][0]))
    dict_rad = '%.2f $\pm$ %.2f'%(wd_mean_rad, wd_std_rad)
    my_dict['Radial Energy'].append(dict_rad)
    
    wd_mean_hits, wd_std_hits = calc_wdist_1d(real_dict["hits"+pp+"_list_r"], fake_dict["hits"+pp+"_list"], tot_shw = shw)
    dict_long = '%.2f $\pm$ %.2f'%(wd_mean_hits, wd_std_hits)
    my_dict['N. Hits'].append(dict_hits)
    
    spinal = np.arange(0.5, 78+0.5)
    wd_mean_long, wd_std_long = calc_wdist_1d(spinal, spinal, uw=real_dict["e_layers"+pp+"_list_r"], vw=fake_dict["e_layers"+pp+"_list"], tot_shw = len(real_dict["e_layers"+pp+"_list_r"]))
    dict_long = '%.2f $\pm$ %.2f'%(wd_mean_long, wd_std_long)
    my_dict['Energy along y'].append(dict_long)
    
    # # hide axes
    # df = pd.DataFrame(my_dict)
    # # df.index = my_dict.keys()
    # filename = my_dir+'Table_wd_'+pp+'_'+str(shw)+ '_showers'
    # filename_tex = filename+'.tex'
    # template = r'''\documentclass{{article}}
    # \usepackage{{booktabs}}
    # \begin{{document}}
    # {}
    # \end{{document}}
    # '''
    # with open(filename_tex, 'wb') as f:
    #     f.write(bytes(template.format(df.to_latex()),'UTF-8'))
    # subprocess.call(['pdflatex', filename_tex])
    # fig, ax = plt.subplots()
    # fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    # ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
    # fig.tight_layout()
    # if threshold ==0: plt.savefig(filename+'.pdf', dpi=100, bbox_inches='tight')
    # else: plt.savefig(filename+'.pdf', dpi=100, bbox_inches='tight')
    
    # log_path = Path(filename+'.log').parent.parent.parent.parent.parent.parent.parent / (Path(filename).name + '.log')
    # log_path.unlink() # remove this file: there are memory information and so on 
    
    # normalized WD
    wd_dictionary['Shower Start Layer mean']= wd_mean_sl / np.std(real_dict["start_layer_list_r"+pp])
    wd_dictionary['Shower Start Layer std']= wd_std_sl / np.std(real_dict["start_layer_list_r"+pp])
    wd_dictionary['Energy Sum mean']= wd_mean_esum / np.std(real_dict["e_sum"+pp+"_list_r"])
    wd_dictionary['Energy Sum std']= wd_std_esum / np.std(real_dict["e_sum"+pp+"_list_r"])
    wd_dictionary['COG x mean'] = wd_mean_cog_x/ np.std(real_dict["cog_x_r"+pp])
    wd_dictionary['COG x std'] = wd_std_cog_x / np.std(real_dict["cog_x_r"+pp])
    wd_dictionary['COG y mean'] = wd_mean_cog_y / np.std(real_dict["cog_y_r"+pp])
    wd_dictionary['COG y std'] = wd_std_cog_y / np.std(real_dict["cog_y_r"+pp])
    wd_dictionary['Visible Cell Energy mean'] = wd_mean_hits / np.std(real_dict["hits"+pp+"_list_r"])
    wd_dictionary['Visible Cell Energy std'] = wd_std_hits / np.std(real_dict["hits"+pp+"_list_r"])
    wd_dictionary['Energy along y mean'] = wd_mean_long / np.std(real_dict["e_layers"+pp+"_list_r"])
    wd_dictionary['Energy along y std'] = wd_std_long / np.std(real_dict["e_layers"+pp+"_list_r"])
    wd_dictionary['# Hits mean'] = wd_mean_hits / np.std(real_dict["hits"+pp+"_list_r"])
    wd_dictionary['# Hits std'] = wd_std_hits / np.std(real_dict["hits"+pp+"_list_r"])
    wd_dictionary['COG z mean'] = wd_mean_cog_z / np.std(real_dict["cog_z_r"+pp])
    wd_dictionary['COG z std'] = wd_std_cog_z / np.std(real_dict["cog_z_r"+pp])
    wd_dictionary['Radial Energy mean'] = wd_mean_rad / np.std(real_dict["e_radial"+pp+"_r"][0])
    wd_dictionary['Radial Energy std'] = wd_std_rad / np.std(real_dict["e_radial"+pp+"_r"][0])
    
    with open(my_dir+'wd.txt', 'w') as file:
        file.write(json.dumps(wd_dictionary))
         
    return wd_dictionary

def kld(a, b ,num_quantiles=25, tot_shw=100, iterations = 5):
    kl = []
    j=0
    batch_size = int(tot_shw/iterations)
    
    np.random.shuffle(a)
    np.random.shuffle(b)
    for _ in range(iterations):
        if len(a[j:j+batch_size]) == 0:
            continue  # or handle it some other way
        quantiles = np.quantile(a[j:j+batch_size], np.linspace(0.,1.,num_quantiles+1))
        quantiles[0] = float('-inf')
        quantiles[-1] = float('inf')
        pk = np.histogram(a[j:j+batch_size], quantiles)[0]/len(a)
        qk = np.histogram(b[j:j+batch_size], quantiles)[0]/len(b)
        kl.append(entropy(pk, qk))
        j += batch_size
    return np.mean(kl), np.std(kl)

def kl_table(real_dict, fake_dict, shw, threshold, my_dir, ecal=False, hcal=False):
    my_dict={'COG x' : [],'COG y' : [],'COG z' : [], 'Shower Start Layer': [], 'Visible Cell Energy': [], 
            'Energy Sum' : [], 'Radial Energy': [],'N. Hits': [], 'Energy along y': []}
    kl_dictionary = {}
        
    if ecal: pp,l = '_ecal', 30
    elif hcal: pp,l = '_hcal', 48
    else: pp,l = '', 78
    kl_mean_cog_x, kl_std_cog_x = kld(real_dict["cog_x_r"+pp]-0.5, fake_dict["cog_x"+pp],num_quantiles=25, tot_shw = shw)
    kl_mean_cog_y, kl_std_cog_y = kld(real_dict["cog_y_r"+pp], fake_dict["cog_y"+pp],num_quantiles=25, tot_shw = shw)
    kl_mean_cog_z, kl_std_cog_z = kld(real_dict["cog_z_r"+pp]-0.5, fake_dict["cog_z"+pp],num_quantiles=25, tot_shw = shw)
    dict_cog_x, dict_cog_y, dict_cog_z = '%.3f $\pm$ %.3f'%(kl_mean_cog_x, kl_std_cog_x), '%.3f $\pm$ %.3f'%(kl_mean_cog_y, kl_std_cog_y), '%.3f $\pm$ %.3f'%(kl_mean_cog_z, kl_std_cog_z)
    my_dict['COG x'].append(dict_cog_x)
    my_dict['COG y'].append(dict_cog_y)
    my_dict['COG z'].append(dict_cog_z)
    
    kl_mean_sl, kl_std_sl = kld(real_dict["start_layer_list_r"+pp], fake_dict["start_layer_list"+pp], num_quantiles=25, tot_shw = shw)
    dict_sl = '%.3f $\pm$ %.3f'%(kl_mean_sl, kl_std_sl)
    my_dict['Shower Start Layer'].append(dict_sl)
    
    kl_mean_hits, kl_std_hits = kld(real_dict["hits"+pp+"_list_r"], fake_dict["hits"+pp+"_list"], num_quantiles=25, tot_shw = shw)
    dict_hits = '%.3f $\pm$ %.3f'%(kl_mean_hits, kl_std_hits)
    my_dict['Visible Cell Energy'].append(dict_hits)
    
    kl_mean_esum, kl_std_esum = kld(real_dict["e_sum"+pp+"_list_r"], fake_dict["e_sum"+pp+"_list"], num_quantiles=25, tot_shw = shw)
    dict_esum = '%.3f $\pm$ %.3f'%(kl_mean_esum, kl_std_esum)
    my_dict['Energy Sum'].append(dict_esum)
    
    kl_mean_rad, kl_std_rad = kld(real_dict["e_radial"+pp+"_r"][0], fake_dict["e_radial"+pp][0], num_quantiles=25, tot_shw = len(real_dict["e_radial"+pp+"_r"]))

    dict_rad = '%.3f $\pm$ %.3f'%(kl_mean_rad, kl_std_rad)
    my_dict['Radial Energy'].append(dict_rad)
    
    kl_mean_hits, kl_std_hits = kld(real_dict["hits"+pp+"_list_r"], fake_dict["hits"+pp+"_list"], num_quantiles=25, tot_shw = shw) 
    dict_hits = '%.3f $\pm$ %.3f'%(kl_mean_hits, kl_std_hits)
    my_dict['N. Hits'].append(dict_hits)
    
    kl_mean_long, kl_std_long = kld(real_dict["e_layers"+pp+"_list_r"], fake_dict["e_layers"+pp+"_list"], num_quantiles=25, tot_shw = len(real_dict["e_layers"+pp+"_list_r"])) 
    dict_long = '%.3f $\pm$ %.3f'%(kl_mean_long, kl_std_long)
    my_dict['Energy along y'].append(dict_long)
    
    # # hide axes
    # df = pd.DataFrame(my_dict)
    # # df.index = my_dict.keys()
    # filename = my_dir+'Table_kl_'+pp+'_'+str(shw)+ '_showers'
    # filename_tex = filename+'.tex'
    # template = r'''\documentclass{{article}}
    # \usepackage{{booktabs}}
    # \begin{{document}}
    # {}
    # \end{{document}}
    # '''
    # with open(filename_tex, 'wb') as f:
    #     f.write(bytes(template.format(df.to_latex()),'UTF-8'))
    # subprocess.call(['pdflatex', filename_tex])
    # fig, ax = plt.subplots()
    # fig.patch.set_visible(False)
    # ax.axis('off')
    # ax.axis('tight')
    # ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
    # fig.tight_layout()
    # if threshold ==0: plt.savefig(filename+'.pdf', dpi=100, bbox_inches='tight')
    # else: plt.savefig(filename+'.pdf', dpi=100, bbox_inches='tight')

    # log_path = Path(filename+'.log').parent.parent.parent.parent.parent.parent.parent / (Path(filename).name + '.log')
    # print(log_path)
    # log_path.unlink() # remove this file: there are memory information and so on
    
    # kl
    kl_dictionary['Shower Start Layer mean'] = kl_mean_sl / np.std(real_dict["start_layer_list_r"+pp])
    kl_dictionary['Shower Start Layer std'] = kl_std_sl / np.std(real_dict["start_layer_list_r"+pp])
    kl_dictionary['Energy Sum mean'] = kl_mean_esum / np.std(real_dict["e_sum"+pp+"_list_r"])
    kl_dictionary['Energy Sum std'] = kl_std_esum / np.std(real_dict["e_sum"+pp+"_list_r"])
    kl_dictionary['COG x mean'] = kl_mean_cog_x/ np.std(real_dict["cog_x_r"+pp])
    kl_dictionary['COG x std'] = kl_std_cog_x / np.std(real_dict["cog_x_r"+pp])
    kl_dictionary['COG y mean'] = kl_mean_cog_y / np.std(real_dict["cog_y_r"+pp])
    kl_dictionary['COG y std'] = kl_std_cog_y / np.std(real_dict["cog_y_r"+pp])
    kl_dictionary['Visible Cell Energy mean'] = kl_mean_hits / np.std(real_dict["hits"+pp+"_list_r"])
    kl_dictionary['Visible Cell Energy std'] = kl_std_hits / np.std(real_dict["hits"+pp+"_list_r"])
    kl_dictionary['Energy along y mean'] = kl_mean_long / np.std(real_dict["e_layers"+pp+"_list_r"])
    kl_dictionary['Energy along y std'] = kl_std_long / np.std(real_dict["e_layers"+pp+"_list_r"])
    kl_dictionary['# Hits mean'] = kl_mean_hits / np.std(real_dict["hits"+pp+"_list_r"])
    kl_dictionary['# Hits std'] = kl_std_hits / np.std(real_dict["hits"+pp+"_list_r"])
    kl_dictionary['COG z mean'] = kl_mean_cog_z / np.std(real_dict["cog_z_r"+pp])
    kl_dictionary['COG z std'] = kl_std_cog_z / np.std(real_dict["cog_z_r"+pp])
    kl_dictionary['Radial Energy mean'] = kl_mean_rad / np.std(real_dict["e_radial"+pp+"_r"][0])
    kl_dictionary['Radial Energy std'] = kl_std_rad / np.std(real_dict["e_radial"+pp+"_r"][0])
    
    with open(my_dir+'kl.txt', 'w') as file:
        file.write(json.dumps(kl_dictionary))
    
    return kl_dictionary
       
def plt_wdPlot(wd_dict, kl_dict, my_dir=None):
    data, data_std, metrics = [], [], []
    data_kl, data_std_kl = [], []
    for k in wd_dict.keys():
        if k.split(" ")[-1] == "mean":
            metrics.append(" ".join(k.split()[:-1]) )
            data.append(wd_dict[k])
            data_kl.append(kl_dict[k])
        elif k.split(" ")[-1] == "std":
            data_std.append(wd_dict[k])
            data_std_kl.append(kl_dict[k]) 
            
    N = len(data)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])
    fig, axs = plt.subplots(figsize=(15, 6), nrows=1, ncols=2, subplot_kw={'projection': 'polar'})
    ax = axs[0]
    ax2 = axs[1]
    # Title 
    ax.set_title("Normalized Wasserstein Distance", y=1.15, fontsize=20)
    ax2.set_title("KL Divergence", y=1.15, fontsize=20)

    for axx in axs:
        axx.set_theta_zero_location("N")
        axx.set_theta_direction(-1)
        axx.set_rlabel_position(90)
        axx.spines['polar'].set_zorder(1)
        axx.spines['polar'].set_color('lightgrey')   

    color_palette = ['#339F00', '#0500FF', '#9CDADB', '#FF00DE', '#FF9900', '#FFFFFF']
    data = np.array(data)
    data_std = np.array(data_std)
    data_kl = np.array(data_kl)
    data_std_kl = np.array(data_std_kl)
    
    ax.errorbar(theta, np.concatenate([data, [data[0]]]), yerr=np.concatenate([data_std, [data_std[0]]]), alpha=0.50, color=color_palette[1])
    ax2.errorbar(theta, np.concatenate([data_kl, [data_kl[0]]]), yerr=np.concatenate([data_std_kl, [data_std_kl[0]]]), alpha=0.50, color=color_palette[0])
    
    ax.fill(theta, np.concatenate([data, [data[0]]]), alpha=0.2, color=color_palette[1])
    ax2.fill(theta, np.concatenate([data_kl, [data_kl[0]]]), alpha=0.2, color=color_palette[0])
    
    ax.fill_between(theta, np.concatenate([data+data_std, [data[0] + data_std[0]]]), y2=np.concatenate([data-data_std, [data[0] - data_std[0]]]), alpha=0.40, color=color_palette[1])
    ax2.fill_between(theta, np.concatenate([data_kl+data_std_kl, [data_kl[0] + data_std_kl[0]]]), y2=np.concatenate([data_kl-data_std_kl, [data_kl[0] - data_std_kl[0]]]), alpha=0.40, color=color_palette[0])
    
    ax.yaxis.set_inverted(True)
    ax2.yaxis.set_inverted(True)
    ax.set_yticks([1, 0.8, 0.6, 0.4, 0.2, 0], ["1", "0.8", "0.6", "0.4", "0.2","0"], color="black", size=8)
    ax2.set_yticks([0.03, 0.02, 0.01, 0], ["0.03", "0.02", "0.01","0"], color="black", size=8)
    
    ax.tick_params(axis='y', which='major', pad=15)
    ax2.tick_params(axis='y', which='major', pad=15)
    ax.tick_params(axis='x', which='major', pad=15)
    ax2.tick_params(axis='x', which='major', pad=15)
    
    ax.set_xticks(theta[:-1], metrics, color='black', size=12)
    ax2.set_xticks(theta[:-1], metrics, color='black', size=12)

    if my_dir is not None:
        plt.savefig(my_dir+'spider2.pdf', dpi=100, bbox_inches='tight')
    plt.close() 
    
def pearson_plot(sim_list, gen_list, labels, names, my_dir=None):
    pearson_sim = np.zeros((len(sim_list), len(sim_list)))
    pearson_gen = np.zeros((len(gen_list), len(gen_list)))
    for i in range(len(sim_list)): 
        for j in range(len(sim_list)):
            mask = np.isfinite(sim_list[i]) & np.isfinite(sim_list[j])
            pearson_sim[i,j] = pearsonr(sim_list[i][mask], sim_list[j][mask]).statistic
            mask = np.isfinite(gen_list[i]) & np.isfinite(gen_list[j]) 
            pearson_gen[i,j] = pearsonr(gen_list[i][mask], gen_list[j][mask]).statistic 
    
    ff, map_color = 10, "BrBG"
    
    fig = plt.figure(10, figsize=(22,6))
    spec = gridspec.GridSpec(1, 3, width_ratios=[2.17, 1.82, 2], wspace=0.1)  
    ax1 = fig.add_subplot(spec[0, 0])
    pearson_sim = np.ma.array(pearson_sim, mask=np.triu(np.ones(pearson_sim.shape), k=1)) # mask out the upper triangle
    im1 = ax1.imshow(pearson_sim, cmap=map_color, vmin= -1, vmax=1, alpha=0.8)
    ax1.set_title(names[0], fontsize=ff)
    ax1.set_xticks(range(len(labels)), labels, rotation=45, fontsize=ff)
    ax1.set_yticks(range(len(labels)), labels, rotation=45, fontsize=ff)
    
    cbar = fig.colorbar(im1, ax=ax1, fraction= 0.042, pad=0.13, orientation='vertical', location='left') #pad=-0.01,
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=ff)
    # cbar.ax.yaxis.set_ticks_position('right')
    
    ax2 = fig.add_subplot(spec[0, 1])
    pearson_gen = np.ma.array(pearson_gen, mask=np.triu(np.ones(pearson_gen.shape), k=1))
    im2 = ax2.imshow(pearson_gen, cmap=map_color, vmin= -1, vmax=1, alpha=0.8)
    ax2.set_title(names[1], fontsize=ff)
    ax2.set_xticks(range(len(labels)), labels, rotation=45, fontsize=ff)
    ax2.set_yticks(range(len(labels)), labels, rotation=45, fontsize=ff)
    
    for i in range(len(sim_list)):
        for j in range(len(sim_list)):
            if i >= j:
                ax1.text(j, i, np.round(pearson_sim[i,j],2),
                        ha="center", va="center", color="k", weight='bold', fontsize=ff-3)
                ax2.text(j, i, np.round(pearson_gen[i,j],2),
                        ha="center", va="center", color="k", weight='bold', fontsize=ff-3)
                
    
    
    diff = np.abs(pearson_sim - pearson_gen)
    ax4 = fig.add_subplot(spec[0, 2])
    diff = np.ma.array(diff, mask=np.triu(np.ones(diff.shape), k=1))
    im4 = ax4.imshow(diff, cmap="YlGn", alpha=0.8)
    ax4.set_title(names[0]+' - '+names[1], fontsize=ff)
    ax4.set_xticks(range(len(labels)), labels, rotation=45, fontsize=ff)
    ax4.set_yticks(range(len(labels)), labels, rotation=45, fontsize=ff)
    
    for i in range(len(sim_list)):
        for j in range(len(sim_list)):
            ax4.text(j, i, np.round(pearson_sim[i,j] - pearson_gen[i,j],2),
                    ha="center", va="center", color="k", weight='bold', fontsize=ff-3)
    
    cbar = fig.colorbar(im4, ax=ax4, fraction=0.042, orientation='vertical', location='right')
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=ff)
    cbar.ax.yaxis.set_ticks_position('right')
     
    if my_dir is not None:
        plt.savefig(my_dir+'pearson.pdf', dpi=100, bbox_inches='tight')
    plt.close()  