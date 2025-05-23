from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import subprocess
import sys
import os
# Add the parent directory of 'utils' to the Python path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from pathlib import Path
import pandas as pd
from matplotlib.widgets import TextBox 

plt.close()
mpl.rcParams['xtick.labelsize'] = 25    
mpl.rcParams['ytick.labelsize'] = 25

mpl.rcParams['font.size'] = 35
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['text.usetex'] = False #True
mpl.rcParams['font.family'] = 'serif'

reco_gen = h5py.File('/data/dust/user/bussthor/public/pion_clouds.h5', 'r')
reco_sim = h5py.File('/data/dust/user/bussthor/public/geant4.h5', 'r')      
dict_pid = {'e-': 11, 'mu-': 13, 'pi+': 211, 'K+' : 321, 'p' : 2212}
# dict_pid = {'unknown': 0, 'e-': 11, 'mu-': 13, 'pi+': 211, 'K+' : 321, 'p' : 2212, 'n' : 2112}
pid_to_index = {pid: i for i, pid in enumerate(dict_pid.values())}
dict_pid_labels = {k: pid_to_index.get(v, -1) for k, v in dict_pid.items()}
print(np.unique(reco_gen['pid'][:].flatten()))
print(np.unique(reco_sim['pid'][:].flatten()))

a, b = 2,3
bins_ = 30
idx = 0 #list(np.arange(0,30)) 
cond = 0 
print(reco_gen['energy'].shape)
print(reco_gen['momentum'].shape)

if cond==0:
    mask_gen = ((reco_gen['energy'][:,idx]>0) & (reco_gen['pid'][:,idx]>0)) & (reco_gen['momentum'][:,idx,1]>-1)
    mask_sim = ((reco_sim['energy'][:,idx]>0) & (reco_sim['pid'][:,idx]>0)) & (reco_sim['momentum'][:,idx,1]>-1)
    print('tot events: ', mask_gen.sum())
    print('tot events: ', mask_sim.sum())
else: 
    my_pid = 0
    idx2 = list(np.arange(0,30)) 
    mask_gen = reco_gen['energy'][:,idx][reco_gen['pid'][:,idx2]==my_pid] >0
    mask_sim = reco_sim['energy'][:,idx][reco_sim['pid'][:,idx2]==my_pid] >0
    # mask_gen = (reco_gen['energy'][:,idx]>0) & (reco_gen['pid'][:,idx]==my_pid) 
    # mask_sim = (reco_sim['energy'][:,idx]>0) & (reco_sim['pid'][:,idx]==my_pid) 
  
# if type(idx) == int:
#     print('idx')
#     n_gen = (mask_gen).sum()
#     n_sim = (mask_sim).sum()
#     plt.title(f'N particles sim: {n_sim} gen {n_gen} ')

def error_plots(gen, sim): 
    reco_gen_err, reco_sim_err = np.sqrt(gen[0]), np.sqrt(sim[0])
    plt.stairs(gen[0] + reco_gen_err, edges=gen[1], baseline=gen[0] - reco_gen_err, color=col, alpha=0.3, fill=True)
    plt.stairs(sim[0] + reco_sim_err, edges=sim[1], baseline=sim[0] - reco_sim_err, color='dimgray', hatch="///")
    
col = 'purple'
plt.figure(1, figsize=(23,15))

if isinstance(idx, int)==False:
    plt.subplot(a, b, 1)
    gen = plt.hist(reco_gen['num_particles'], bins=30 , histtype='step', label='CaloHadronic', color=col, linewidth=3)
    sim = plt.hist(reco_sim['num_particles'], bins=30 , alpha=0.5, label='Geant4', color='gray')
    error_plots(gen, sim) 
    plt.xlabel('$\#$ PFOs')
    plt.yscale('log')
    plt.legend(fontsize=30)

    plt.subplot(a, b, 2)
    if cond==1:
        # en of the 2nd most energetic particle is the 1st particle was a Pion+
        en_gen = reco_gen['energy'][:,idx][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        en_sim = reco_sim['energy'][:,idx][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        my_pid=-1
        en_gen = reco_gen['energy'][:,idx][mask_gen]
        en_sim = reco_sim['energy'][:,idx][mask_sim] 
        
    gen = plt.hist(en_gen, bins=bins_ , histtype='step', label='CaloHadronic', color=col, linewidth=3)
    sim = plt.hist(en_sim, bins=bins_ , alpha=0.5, label='sim', color='gray')
    error_plots(gen, sim) 
    plt.xlabel('Energy [GeV]')
    plt.yscale('log')
    # plt.legend()

    plt.subplot(a, b, 3)
    if cond==1:
        m_gen_0 = reco_gen['momentum'][:,idx,0][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        m_sim_0 = reco_sim['momentum'][:,idx,0][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        m_gen_0 = reco_gen['momentum'][:,idx,0][mask_gen]
        m_sim_0 = reco_sim['momentum'][:,idx,0][mask_sim]
    gen = plt.hist(m_gen_0, bins=bins_ , histtype='step', label='gen', color=col, linewidth=3)
    sim = plt.hist(m_sim_0, bins=bins_ , alpha=0.5, label='sim', color='gray')
    error_plots(gen, sim) 
    plt.xlabel('$p_{x}$ [GeV]')
    plt.yscale('log')
    # plt.legend()

    plt.subplot(a, b, 4)
    if cond==1:
        m_gen_1 = reco_gen['momentum'][:,idx,1][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        m_sim_1 = reco_sim['momentum'][:,idx,1][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        m_gen_1 = reco_gen['momentum'][:,idx,1][mask_gen]
        m_sim_1 = reco_sim['momentum'][:,idx,1][mask_sim]
    gen = plt.hist(m_gen_1, bins=bins_ , histtype='step', color=col, linewidth=3)
    sim = plt.hist(m_sim_1, bins=bins_ , alpha=0.5, color='gray')
    error_plots(gen, sim) 
    plt.xlabel('$p_{y}$  [GeV]')
    plt.yscale('log')

    plt.subplot(a, b, 5)
    if cond==1:
        m_gen_2 = reco_gen['momentum'][:,idx,2][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        m_sim_2 = reco_sim['momentum'][:,idx,2][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        m_gen_2 = reco_gen['momentum'][:,idx,2][mask_gen]
        m_sim_2 = reco_sim['momentum'][:,idx,2][mask_sim]
    gen = plt.hist(m_gen_2, bins=bins_ , histtype='step', color=col, linewidth=3)
    sim = plt.hist(m_sim_2, bins=bins_ , alpha=0.5, color='gray')
    error_plots(gen, sim) 
    plt.xlabel('$p_{z}$  [GeV]')
    plt.yscale('log')

    plt.subplot(a, b, 6)
    if cond==1:
        pid_gen = reco_gen['pid'][:,idx][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        pid_sim = reco_sim['pid'][:,idx][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        pid_gen = reco_gen['pid'][:,idx][mask_gen]
        pid_sim = reco_sim['pid'][:,idx][mask_sim]
    mapped_values_gen = np.array([pid_to_index.get(pid, -1) for pid in pid_gen])
    mapped_values_sim = np.array([pid_to_index.get(pid, -1) for pid in pid_sim])
    gen = plt.hist(mapped_values_sim, bins=np.arange(-0.5, len(pid_to_index)+0.5, 1), range = [0, mapped_values_sim.max()+1], alpha=0.5, color='gray')
    sim = plt.hist(mapped_values_gen, bins=np.arange(-0.5, len(pid_to_index)+0.5, 1), range = [0, mapped_values_gen.max()+1], histtype='step', color=col, linewidth=3)
    plt.ylim([1e0, 1e3])
    error_plots(gen, sim) 
    plt.yscale('log')
    tick_positions = list(dict_pid_labels.values())
    tick_labels = list(dict_pid_labels.keys())
    plt.xticks(tick_positions, tick_labels, rotation=45)
    # plt.xlabel('PID')
    # plt.legend()

    if my_pid==0: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_pid0.png"
    else: 
        if idx==0: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_1mostEnergeticPart.pdf"
        elif idx==1:
            if cond==1: 
                if my_pid==0: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_pid0.png"
                else: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_2mostEnergeticPart_condpi+.png"
            else: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_2mostEnergeticPart.png"
        elif idx==2: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_3mostEnergeticPart.png"
        else: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_all.pdf"

else:
    plt.subplot(a, b, 1)
    if cond==1:
        # en of the 2nd most energetic particle is the 1st particle was a Pion+
        en_gen = reco_gen['energy'][:,idx][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        en_sim = reco_sim['energy'][:,idx][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        my_pid=-1
        en_gen = reco_gen['energy'][:,idx][mask_gen]
        en_sim = reco_sim['energy'][:,idx][mask_sim] 
        
    gen = plt.hist(en_gen, bins=bins_ , histtype='step', label='CaloHadronic', color=col, linewidth=3)
    sim = plt.hist(en_sim, bins=bins_ , alpha=0.5, label='Geant4', color='gray')
    error_plots(gen, sim) 
    plt.xlabel('Energy [GeV]')
    plt.yscale('log')
    
    ax = plt.subplot(a, b, 2)
    
    props = dict(boxstyle='round', edgecolor=col, facecolor='white', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.1, 0.95, ' leading PFO ', transform=ax.transAxes,
            verticalalignment='top', fontsize=40, fontfamily='serif', bbox=props) 
    plt.axis('off')

    plt.subplot(a, b, 3)
    if cond==1:
        m_gen_0 = reco_gen['momentum'][:,idx,0][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        m_sim_0 = reco_sim['momentum'][:,idx,0][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        m_gen_0 = reco_gen['momentum'][:,idx,0][mask_gen]
        m_sim_0 = reco_sim['momentum'][:,idx,0][mask_sim]
    gen = plt.hist(m_gen_0, bins=bins_ , histtype='step', label='gen', color=col, linewidth=3)
    sim = plt.hist(m_sim_0, bins=bins_ , alpha=0.5, label='sim', color='gray')
    error_plots(gen, sim) 
    plt.xlabel('$p_{x}$ [GeV]')
    plt.yscale('log')
    # plt.legend()

    plt.subplot(a, b, 4)
    if cond==1:
        m_gen_1 = reco_gen['momentum'][:,idx,1][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        m_sim_1 = reco_sim['momentum'][:,idx,1][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        m_gen_1 = reco_gen['momentum'][:,idx,1][mask_gen]
        m_sim_1 = reco_sim['momentum'][:,idx,1][mask_sim]
    gen = plt.hist(m_gen_1, bins=bins_ , histtype='step', color=col, linewidth=3, label='CaloHadronic')
    sim = plt.hist(m_sim_1, bins=bins_ , alpha=0.5, color='gray', label='Geant4')
    error_plots(gen, sim) 
    plt.xlabel('$p_{y}$ [GeV]')
    plt.yscale('log')
    plt.legend(fontsize=30, loc='upper left')

    plt.subplot(a, b, 5)
    if cond==1:
        m_gen_2 = reco_gen['momentum'][:,idx,2][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        m_sim_2 = reco_sim['momentum'][:,idx,2][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        m_gen_2 = reco_gen['momentum'][:,idx,2][mask_gen]
        m_sim_2 = reco_sim['momentum'][:,idx,2][mask_sim]
    gen = plt.hist(m_gen_2, bins=bins_ , histtype='step', color=col, linewidth=3)
    sim = plt.hist(m_sim_2, bins=bins_ , alpha=0.5, color='gray')
    error_plots(gen, sim) 
    plt.xlabel('$p_{z}$ [GeV]')
    plt.yscale('log')

    plt.subplot(a, b, 6)
    if cond==1:
        pid_gen = reco_gen['pid'][:,idx][reco_gen['pid'][:,idx2]==my_pid][mask_gen]
        pid_sim = reco_sim['pid'][:,idx][reco_sim['pid'][:,idx2]==my_pid][mask_sim]
    else: 
        pid_gen = reco_gen['pid'][:,idx][mask_gen]
        pid_sim = reco_sim['pid'][:,idx][mask_sim]
    mapped_values_gen = np.array([pid_to_index.get(pid, -1) for pid in pid_gen])
    mapped_values_sim = np.array([pid_to_index.get(pid, -1) for pid in pid_sim])
    gen =plt.hist(mapped_values_sim, bins=np.arange(-0.5, len(pid_to_index)+0.5, 1), range = [0, mapped_values_sim.max()+1], alpha=0.5, color='gray')
    sim = plt.hist(mapped_values_gen, bins=np.arange(-0.5, len(pid_to_index)+0.5, 1), range = [0, mapped_values_gen.max()+1], histtype='step', color=col, linewidth=3)
    # plt.ylim([1e0, 1e3])
    error_plots(gen, sim) 
    plt.yscale('log')
    tick_positions = list(dict_pid_labels.values())
    tick_labels = list(dict_pid_labels.keys())
    plt.xticks(tick_positions, tick_labels, rotation=45)
    # plt.xlabel('PID')
    # plt.legend()

    if my_pid==0: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_pid0.png"
    else: 
        if idx==0: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_1mostEnergeticPart.pdf"
        elif idx==1:
            if cond==1: 
                if my_pid==0: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_pid0.png"
                else: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_2mostEnergeticPart_condpi+.png"
            else: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_2mostEnergeticPart.png"
        elif idx==2: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_3mostEnergeticPart.png"
        else: folder = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_all.pdf"
 

print(folder)
plt.savefig(folder)
plt.close()
