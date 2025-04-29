from tqdm import tqdm
import numpy as np
# import torch.utils.data
# from torch import optim
import h5py
import time
import matplotlib
import matplotlib.pyplot as plt
import sys
# from models.shower_flow import compile_HybridTanH_model
# import logging
import os
from utils.misc import Config, Configs
from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/configs_sf.yaml')
dataset_path = cfg.dataset_path
print(dataset_path)

def _calculate_points_per_layer(events, axis=1, bins=78, is_hcal = False, is_ecal = False):
    points_per_layer= []
    for event in events:
        num_p, _ = np.histogram(event[:, axis][event[:, 3] > 0], bins=bins, range=(0, bins))
        if is_ecal: num_p = num_p[:30]
        if is_hcal: num_p = num_p[30:] 
        points_per_layer.append(num_p)
    return np.array(points_per_layer)

def _calculate_energy_per_layer(events, axis=1, bins=78, is_hcal = False, is_ecal = False):
    energy_per_layer  = []
    for event in events:
        en_p, _ = np.histogram(event[:, axis][event[:, 3] > 0], bins=bins, range=(0, bins), weights=event[:, 3][event[:, 3] > 0]) 
        if is_hcal: en_p = en_p[30:]
        if is_ecal: en_p = en_p[:30] 
        energy_per_layer.append(en_p)    
    return np.array(energy_per_layer)

def _calculate_n_points(events):
    return np.sum(events[:,3]>0, axis=1)
    
n_files = 100
tot_showers = n_files * 20000

# data
to_save_energy = np.zeros((tot_showers, 1))
to_save_n_points = np.zeros((tot_showers, 1))
to_save_points_per_layer = np.zeros((tot_showers, 78))
to_save_energy_per_layer = np.zeros((tot_showers, 78))

max_n_of_points, max_n_of_points_clust = 0, 0
MAX_LEN = 13000
CELLSIZE = 6
counts_entries = 0
energy_sum_check = 0

for file_idx in tqdm(range(1, n_files+1)): # 11 for old data 
    f = h5py.File(dataset_path.format(file_idx), 'r')['events'][:]
    energy = h5py.File(dataset_path.format(file_idx), 'r')['energy'][:] #.reshape(-1)

    print('loaded file %d | '%(file_idx), end=" ")
    n_points = _calculate_n_points(f)
    points_per_l = _calculate_points_per_layer(np.moveaxis(f,-1,-2))
    en_per_l = _calculate_energy_per_layer(np.moveaxis(f,-1,-2))
     
    events_per_file = f.shape[0] 
    if file_idx==1:
        idx_start, idx_end = 0, events_per_file  
    else:
        idx_start = idx_end
        idx_end = idx_end+events_per_file
        
    print(idx_start, idx_end)
    counts_entries += events_per_file
    to_save_energy[idx_start:idx_end] = energy 
    to_save_energy_per_layer[idx_start:idx_end] = en_per_l
    to_save_n_points[idx_start:idx_end] = np.expand_dims(n_points, axis=1)
    to_save_points_per_layer[idx_start:idx_end] = points_per_l


to_save_energy = to_save_energy[:counts_entries]
to_save_n_points = to_save_n_points[:counts_entries]
to_save_points_per_layer = to_save_points_per_layer[:counts_entries]
to_save_energy_per_layer = to_save_energy_per_layer[:counts_entries]

name = 'HGx9'
outfile_energy = cfg.load_dir+'energy_for_SF_'+name+'.npy'
my_dir = cfg.load_dir+'points_'+name
os.makedirs(my_dir, exist_ok=True)

print('dir:', my_dir)
outfile_n_points         = my_dir+'/n_points_for_SF.npy'
outfile_n_points_per_l   = my_dir+'/n_points_per_layer_for_SF.npy'
outfile_n_energy_per_l   = my_dir+'/energy_per_layer_for_SF.npy'

np.save(outfile_energy, to_save_energy)
np.save(outfile_n_points, to_save_n_points)
np.save(outfile_n_points_per_l, to_save_points_per_layer)
np.save(outfile_n_energy_per_l, to_save_energy_per_layer)
