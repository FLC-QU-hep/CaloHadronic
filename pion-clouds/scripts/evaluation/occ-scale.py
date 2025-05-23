from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import subprocess
import sys
import os
# Add the parent directory of 'utils' to the Python path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.plotting import cfg as cfg_plt
from utils.projection import projection_ecal_hcal
from utils.plotting_showers import *
from utils.plotting_features import *
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
from memory_profiler import profile

plt.close()
mpl.rcParams['xtick.labelsize'] = 25    
mpl.rcParams['ytick.labelsize'] = 25

mpl.rcParams['font.size'] = 35
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['text.usetex'] = False #True
mpl.rcParams['font.family'] = 'serif'

name='HGx9_CNF_ECAL_HCAL_30'
load_dataset_and_project = False
compute_features = False
load_array_and_plot = True

both_to_save = True
fake_SHOWERS_TO_SAVE, real_SHOWERS_TO_SAVE =  False, True

shw = 50002
e_min, e_max = 10, 90 #49,51 #18, 22 #84,86 #13,17
print('e_min:', e_min, 'e_max: ', e_max)
 
if both_to_save: fake_SHOWERS_TO_SAVE, real_SHOWERS_TO_SAVE = True, True
print(fake_SHOWERS_TO_SAVE, real_SHOWERS_TO_SAVE) 
if e_max<18: inc_en = '15GeV_'
elif (e_max<24) & (e_min>15): inc_en = '20GeV_' 
elif (e_max<54) & (e_min>45): inc_en = '50GeV_'
elif (e_max<89) & (e_min>80): inc_en = '85GeV_'
else: inc_en = '' #default 
# inc_en = inc_en + 'noSF_' 

print(name)
if name.split('_')[2]=='onlyHCAL': only_hcal_flag = True
else: only_hcal_flag = False
print('only_hcal_flag ', only_hcal_flag)

if load_dataset_and_project:
    dataset_path = '/data/dust/user/mmozzani/pion-clouds/dataset/hdf5_HGx9/all_interactions_pions_regular_ECAL+HCAL_10-90GeV_{}.slcio.root_with_time.hdf5' 

    if fake_SHOWERS_TO_SAVE:    
        fake_showers = h5py.File('/data/dust/user/mmozzani/pion-clouds/files/generated_showers/'+name+'/gen_showers_'+inc_en+str(shw)+'.hdf5', 'r')['events'][:]   
        fake_inc_energies = h5py.File('/data/dust/user/mmozzani/pion-clouds/files/generated_showers/'+name+'/gen_showers_'+inc_en+str(shw)+'.hdf5', 'r')['energy'][:]   
        fake_inc_energies = (fake_inc_energies+1)/2*100   
        shw = fake_showers.shape[0] #shww
        print(shw) 
        if only_hcal_flag: 
            mask = fake_showers[:, 1] < 30 
            for j in range(4): fake_showers[:, j][mask] = 0  
             
    if real_SHOWERS_TO_SAVE:       
        real_showers  = np.zeros((shw, 4, 5000)) #np.zeros((fake_showers.shape[0], fake_showers.shape[1], 5000)) #fake_showers.shape[2])) #10000)) 
        point_per_layer_file  = np.zeros((shw, 5000))
        inc_energies = np.zeros((shw, 1))
        files = 3
        for i in tqdm(range(files)): 
            if name.split('_')[0]=='HGx9': file_idx = np.random.randint(1, 100, size=files) #[50] # #[1]
            else: file_idx = [2, 4, 7, 9]
            shw_taken_per_file = int(shw/files)
            f = h5py.File(dataset_path.format(file_idx[i]), 'r')['events'][:] 
            energy = h5py.File(dataset_path.format(file_idx[i]), 'r')['energy'][:] 
            energy_mask = (energy.reshape(-1) > e_min) & (energy.reshape(-1) < e_max)
            idx_sorted = np.argsort(energy[energy_mask].reshape(-1))
            idx_showers = np.linspace(1, energy_mask.sum()-1, shw_taken_per_file).astype(int)
            # idx = np.sort((np.random.rand(1, shw_taken_per_file) * energy_mask.sum())).astype(int)[0]   
            idx_start = shw_taken_per_file * i 
            idx_end = shw_taken_per_file * (i+1)
            inc_energies[idx_start:idx_end] = energy[energy_mask][idx_sorted][idx_showers]
            real_showers[idx_start:idx_end] = f[energy_mask][idx_sorted][idx_showers,:4]
            # point_per_layer_file[idx_start:idx_end] = Nperlayer[idx ] 
        del f 
        # max_len = (real_showers[:, 3] > 0).sum(axis=1).max()
        # real_showers = real_showers[:, :, -max_len:]   
        # if name.split('_')[0]=='HGx9': real_showers[:,[0,1,2,3],:] = real_showers[:,[0,2,1,3],:]

        # sorting real_shw per number of points (fake ones are already sorted))
        idx_sorted_real = np.argsort(np.sum(real_showers[:,3]>0, axis=1))
        real_showers = real_showers[idx_sorted_real]
        inc_energies = inc_energies[idx_sorted_real]
        real_showers[:,3,:] = real_showers[:,3,:]*1000  
    
Ymin, Ymin_hcal, Ymax = 0, 30, 78
cell_size_ecal = 5
cell_size_hcal = 30

Xmin, Xmax = -450, 450
Zmin, Zmax = -450, 450 

x_bins = np.arange(Xmin, Xmax+cell_size_ecal, cell_size_ecal)
z_bins = np.arange(Zmin, Zmax+cell_size_ecal, cell_size_ecal)
x_bins_hcal = np.arange(Xmin, Xmax+cell_size_hcal, cell_size_hcal)
z_bins_hcal = np.arange(Zmin, Zmax+cell_size_hcal, cell_size_hcal)
y_bins = np.arange(Ymin, Ymax+1, 1)
y_bins_ecal = np.arange(Ymin, Ymin_hcal+1, 1)
y_bins_hcal = np.arange(Ymin_hcal, Ymax+1, 1)

os.makedirs('/data/dust/user/mmozzani/pion-clouds/figs/occ-scale/'+name, exist_ok=True)
directory = '/data/dust/user/mmozzani/pion-clouds/figs/occ-scale/'+name+'/'

#save arrays
os.makedirs('/data/dust/user/mmozzani/pion-clouds/files/projected_array/'+name, exist_ok=True)
os.makedirs('/data/dust/user/mmozzani/pion-clouds/files/projected_array/'+name+'/'+inc_en+str(shw), exist_ok=True)
save_dir_fake = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/'+name+'/'+inc_en+str(shw) 

os.makedirs('/data/dust/user/mmozzani/pion-clouds/files/projected_array/Geant4/', exist_ok=True)
os.makedirs('/data/dust/user/mmozzani/pion-clouds/files/projected_array/Geant4/'+inc_en+str(shw), exist_ok=True)
save_dir_real = '/data/dust/user/mmozzani/pion-clouds/files/projected_array/Geant4/'+inc_en+str(shw) 

os.makedirs(directory+'Features_/', exist_ok=True)
os.makedirs(directory+'Features_/'+inc_en+str(shw), exist_ok=True)
directory_2 = directory+'Features_/'+inc_en+str(shw)+'/'
print(directory_2 )
l = 0 

if load_dataset_and_project: 
    if fake_SHOWERS_TO_SAVE: np.save(save_dir_fake+'/inc_en.npy', fake_inc_energies)
    if real_SHOWERS_TO_SAVE: np.save(save_dir_real+'/inc_en.npy', inc_energies)
    
    if both_to_save: 
        print('saving both real and fake showers...')
        showers = [fake_showers, real_showers]
        inc_energy_ = [fake_inc_energies, inc_energies]
        save_dir = [save_dir_fake, save_dir_real]
        # del fake_showers, real_showers
        fake_showers, real_showers = showers[0], showers[1]
        plot_not_proj(directory, real_showers, fake_showers)
        plotting_pionclouds(directory, real_showers, fake_showers, only_hcal=only_hcal_flag)
        print('not proj plots done')

    elif fake_SHOWERS_TO_SAVE: 
        print('saving fake showers...')
        showers, save_dir = [fake_showers], [save_dir_fake]
        inc_energy_ = [fake_inc_energies]
        del fake_showers    
    elif real_SHOWERS_TO_SAVE:
        print('saving real showers...')
        showers, save_dir = [real_showers], [save_dir_real]
        inc_energy_ = [inc_energies]
        del real_showers 
    
    for gg in range(len(showers)):    
        bs = 5000
        l0, l1, l2, l3 = np.array(showers[gg]).shape[0], int(x_bins.shape[0] -1), int(y_bins.shape[0] -1), int(z_bins.shape[0] -1)  
        file = h5py.File(save_dir[gg]+'/events_'+str(l0)+'.hdf5', "w")
        events_goup = file.create_dataset("projected_showers", (l0, l1, l2, l3), dtype = "f4")
        energy_goup = file.create_dataset("inc_energy", (l0, 1), dtype = "f4")
        
        for chunck in range(0, shw, bs): # do it in chuncks
            if (shw - chunck) < bs: bs = shw - chunck   
            print(f"Writing to: events_goup[{chunck}:{chunck+bs}], shape: {events_goup[chunck:chunck+bs].shape}")
            print(f"Data shape: {showers[gg].shape}")
            events = projection_ecal_hcal(showers[gg][chunck:chunck+bs], x_bins, x_bins_hcal, y_bins, y_bins_ecal, y_bins_hcal, z_bins, z_bins_hcal)
            events_goup[chunck:chunck+bs] = np.ascontiguousarray(events).astype('float32')
            energy_goup[chunck:chunck+bs] = np.ascontiguousarray(inc_energy_[gg][chunck:chunck+bs]).astype('float32') 
            del events
        file.close()

# @profile 
def get_features(events, all = True, num_showers = 20000, thr = 0.25):
    occ = 0
    occ_list, occ_list_025 = [], []# occupancy
    hits_list = [] # energy per cell
    e_layers_list, e_layers_std_list, e_radial, e_sum_list = [], [], [], []
    e_sum_hcal_list, e_sum_ecal_list, e_radial_ecal, e_radial_hcal = [], [], [], []
    e_layers_ecal_list, e_layers_hcal_list =[], []
    hits_ecal_list, hits_hcal_list = [], []
    start_layer_list = []
    X, Z, Y, numb_active_hits_tot, numb_active_hits_ecal, numb_active_hits_hcal = [],[], [], [], [], []   
    hits_ecal, hits_hcal = 0, 0   
    if all: tot = np.array(events).shape[0]
    else: tot = int(num_showers)    
    for j in tqdm(range(tot), mininterval = 10, desc ="Features..."):
        occ, occ_025, e_sum, e_sum_hcal, e_sum_ecal  = 0, 0, 0, 0, 0 
        my_shw = np.moveaxis(np.array(events[j]), 0, 1) 
        numb_active_hits_tot.append(len(np.where(my_shw>0)[0]))
        numb_active_hits_ecal.append(len(np.where(my_shw[:30]>0)[0]))
        numb_active_hits_hcal.append(len(np.where(my_shw[30:]>0)[0]))
        e_layers, e_layers_ecal, e_layers_hcal = [], [], [] 
        for l, layer in enumerate(my_shw): 
            # layer = my_shw[l] # *1000  energy rescale not needed, already in MeV
            layer[layer < thr] = 0
            hit_mask = layer > 0   
                             
            occ += hit_mask.sum()
            occ_025 += (layer > 0.25).sum()
            Y.append(np.repeat(l, hit_mask.sum()))
            layer_hits = layer[hit_mask]
            e_sum += layer.sum()
            if l<30: e_sum_ecal += layer.sum()
            else: e_sum_hcal += layer.sum()
            
            # get start layer 
            if l == 0: start_l, e_sum_old = l, layer.sum()
            elif layer.sum() > e_sum_old: start_l, e_sum_old = l, layer.sum()
            ######################################### 
            
            e_layers.append(layer.sum())
            hits_list.append(layer_hits)
            
            if l>29:
                e_layers_hcal.append(layer.sum()) 
                layer_hcal_hits = layer[hit_mask]  
                hits_ecal_list.append(layer_hits) 
            else:
                e_layers_ecal.append(layer.sum()) 
                layer_ecal_hits = layer[hit_mask] 
                hits_hcal_list.append(layer_hits)
    
            # get radial profile ####################### 
            x_hit_idx, z_hit_idx = np.where(hit_mask)  
            e_cell = layer[x_hit_idx, z_hit_idx]  
            center_of_cell = 89.5 #cell_size_ecal/2          
            # if only_hcal_flag: center_of_cell = 15
            dist_to_origin = np.sqrt((x_hit_idx-center_of_cell)**2 + (z_hit_idx-center_of_cell )**2)
            X.append(x_hit_idx) 
            Z.append(z_hit_idx) 
            e_radial.append([dist_to_origin, e_cell]) 
            if l<30: e_radial_ecal.append([dist_to_origin, e_cell])
            else: e_radial_hcal.append([dist_to_origin, e_cell]) 
            ############################################  
        start_layer_list.append(start_l)
        e_layers_list.append(e_layers)
        e_layers_ecal_list.append(e_layers_ecal)
        e_layers_hcal_list.append(e_layers_hcal)
        occ_list.append(occ)
        occ_list_025.append(occ_025)
        e_sum_list.append(e_sum)
        e_sum_ecal_list.append(e_sum_ecal)
        e_sum_hcal_list.append(e_sum_hcal) 
              
    e_radial = np.concatenate(e_radial, axis=1)
    e_radial_ecal = np.concatenate(e_radial_ecal, axis=1)
    e_radial_hcal = np.concatenate(e_radial_hcal, axis=1)
    occ_list = np.array(occ_list)
    e_sum_list = np.array(e_sum_list)
    start_layer_list = np.array(start_layer_list)
    e_sum_ecal_list = np.array(e_sum_ecal_list)
    e_sum_hcal_list = np.array(e_sum_hcal_list)
    hits_list = np.concatenate(hits_list)
    hits_ecal_list = np.concatenate(hits_ecal_list)
    hits_hcal_list = np.concatenate(hits_hcal_list)
    e_layers_list = np.array(e_layers_list).sum(axis=0)/tot 
    e_layers_ecal_list = np.array(e_layers_ecal_list).sum(axis=0)/tot
    e_layers_hcal_list = np.array(e_layers_hcal_list).sum(axis=0)/tot
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Z = np.concatenate(Z) 
    return e_radial, e_radial_ecal, e_radial_hcal, occ_list, occ_list_025, e_sum_list, e_sum_ecal_list, e_sum_hcal_list, hits_list, hits_ecal_list, hits_hcal_list, e_layers_list, e_layers_ecal_list, e_layers_hcal_list, X, Y, Z, numb_active_hits_tot, numb_active_hits_ecal, numb_active_hits_hcal, start_layer_list
    
def start_layer_events(events, tot=shw):
    shower_start_layer = []
    for j in range(tot):
        max_energetic_point = np.argmax(events[j])
        layer_of_max_energetic_point = events[j][max_energetic_point][1]
        shower_start_layer.append(layer_of_max_energetic_point) 
    return shower_start_layer

             
f_names = ['e_radial','e_radial_ecal', 'e_radial_hcal', 'occ_list', 'occ_list_025', 'e_sum_list', 
             'e_sum_ecal_list', 'e_sum_hcal_list', 'hits_list', 'hits_ecal_list', 'hits_hcal_list', 'e_layers_list', 
             'e_layers_ecal_list', 'e_layers_hcal_list',  'X', 'Y', 'Z', 'numb_active_hits_list', 'numb_active_hits_ecal', 
             'numb_active_hits_hcal', 'start_layer_list','cog_x', 'cog_y', 'cog_z']
f_names_r=['e_radial_r','e_radial_ecal_r', 'e_radial_hcal_r', 'occ_list_r', 'occ_list_025_r', 'e_sum_list_r', 
             'e_sum_ecal_list_r', 'e_sum_hcal_list_r', 'hits_list_r', 'hits_ecal_list_r', 'hits_hcal_list_r', 'e_layers_list_r',
             'e_layers_ecal_list_r', 'e_layers_hcal_list_r', 'X_r', 'Y_r', 'Z_r', 'numb_active_hits_list_r', 'numb_active_hits_ecal_r', 
             'numb_active_hits_hcal_r', 'start_layer_list_r', 'cog_x_r', 'cog_y_r', 'cog_z_r']
threshold = 1e-2

if compute_features:
    k = shw
    
    if l!=0: k = np.array(l).shape[0]
    print('loading...')
    if fake_SHOWERS_TO_SAVE: 
        events = h5py.File(save_dir_fake+'/events_'+str(k)+'.hdf5', 'r')['projected_showers'][:]
        print('loading gen done')
    if real_SHOWERS_TO_SAVE: 
        events_r = h5py.File(save_dir_real+'/events_'+str(k)+'.hdf5', 'r')['projected_showers'][:]
        print('loading sim done')
         
    if (both_to_save) & (shw<15000):
        # plotting_correlations(directory_2, events.astype(np.float16), events_r.astype(np.float16), axis=0)
        # print('correlation plots done')
        plotting_correlations_withCOGy(directory_2, events.astype(np.float16), events_r.astype(np.float16))
        print('correlation cogy plots done')
        plotting_correlations_withN(directory_2, events.astype(np.float16), events_r.astype(np.float16))
        print('correlation n hits plots done')
        sys.exit()
        # directory_shw = directory_2+'3Dplots/'
        # os.makedirs(directory_shw, exist_ok=True)
        # i = int(shw/2) 
        # ev, ev_names, ev_title = [], [], []
        # for i in range(shw): #np.arange(i, i+20, 1): 
        #     ev.append(events[i].astype(np.float16))
        #     ev_names.append("3d_shower_fake_"+str(i))
        #     ev_title.append("CaloHadronic")
        # # plt_3dShowers([events_r[i+1].astype(np.float16)], model_titles=["Geant4"], save_titles=["3d_shower_real"], my_dir=directory_shw)
        # plt_3dShowers([events_r[i].astype(np.float16)] + ev, model_titles=["Geant4"] + ev_title, save_titles=["3d_shower_real"] + ev_names, my_dir=directory_shw)
        # print('3d shower plot done') 
        # sys.exit()

    print('Getting features...')

    if fake_SHOWERS_TO_SAVE:
        mylist = get_features(events, all=False, num_showers = shw, thr = threshold)
        for k, var in enumerate(mylist): 
            np.save(save_dir_fake+'/'+str(f_names[k])+'.npy', var)
        del mylist 
        print('features done')
         
        if shw>3000:
            bs = 1000
            del events 
            ev = h5py.File(save_dir_fake+'/events_'+str(shw)+'.hdf5', 'r')['projected_showers']
            print('loading in batches...')
            c = [] 
            for j in tqdm(range(0, ev.shape[0], bs)):
                if (j+bs) > ev.shape[0]: 
                    bs = ev.shape[0] - j
                    
                events = ev[j:(j+bs)] 
                c.append(get_cog2(events.astype(np.float32)))    
            my_list_cog = np.concatenate(c, axis=1) 
            
        else:
            my_list_cog = get_cog2(events.astype(np.float32)) 
        print('cog done') 
        
        del events 
        # mylist = tuple(mylist) + tuple(my_list_cog) 
        mylist = tuple(my_list_cog)
        cog_names = ['cog_x', 'cog_y', 'cog_z']
        for k, var in enumerate(mylist): 
            np.save(save_dir_fake+'/'+str(cog_names[k])+'.npy', var)
    
    if real_SHOWERS_TO_SAVE:
        mylist_r = get_features(events_r, all=False, num_showers = shw, thr = threshold)
        for k, var in enumerate(mylist_r): 
            np.save(save_dir_real+'/'+str(f_names_r[k])+'.npy', var)
        del mylist_r
        print('features done')
        
        if shw>3000:
            bs = 1000
            del events_r
            ev = h5py.File(save_dir_real+'/events_'+str(shw)+'.hdf5', 'r')['projected_showers']
            print('loading in batches...')
            c = [] 
            for j in tqdm(range(0, ev.shape[0], bs)):
                if (j+bs) > ev.shape[0]: 
                    bs = ev.shape[0] - j
                    
                events = ev[j:(j+bs)] 
                c.append(get_cog2(events.astype(np.float32)))    
            my_list_cog_r = np.concatenate(c, axis=1) 
            
        else:
            my_list_cog_r = get_cog2(events_r.astype(np.float32))
            
        # mylist_r = tuple(mylist_r) + tuple(my_list_cog_r)
        mylist_r = tuple(my_list_cog_r)
        cog_names_r = ['cog_x_r', 'cog_y_r', 'cog_z_r']
        for k, var in enumerate(mylist_r): 
            np.save(save_dir_real+'/'+str(cog_names_r[k])+'.npy', var) 

def shower_for_radial_in_mm(radial_ecal, radial_hcal):
    # ecal has cell sizes of 5 mm while hcal of 30 mm 
    radial_ecal = radial_ecal * 5
    radial_hcal = radial_hcal * 30
    return np.concatenate((radial_ecal, radial_hcal), axis=1)
    
    
print('SAVE DONE')
if load_array_and_plot:
    print('LOADING...')
    f_names = f_names + ['inc_en']  
    f_names_r = f_names_r + ['inc_en'] 
    
    real_dict = {}    
    for k, name in enumerate(f_names_r): 
        real_dict[name] = np.load(save_dir_real+'/'+name+'.npy')
         
    fake_dict = {}
    for k, name in enumerate(f_names): 
        fake_dict[name] = np.load(save_dir_fake+'/'+name+'.npy')     
    
    b=29
    bins_y = 78
    plt.clf()
    my_model_label = "CaloHadronic"
    num_of_shw = len(real_dict["e_sum_list_r"])
      
    plt_spinal(real_dict["e_layers_list_r"], [fake_dict["e_layers_list"]], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2)
    print('done spinal')
    plt_radial(real_dict["e_radial_r"], [fake_dict["e_radial"]], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2)
    print('done radial')
    fake_rad_mm = shower_for_radial_in_mm(fake_dict["e_radial_ecal"], fake_dict["e_radial_hcal"])
    real_rad_mm = shower_for_radial_in_mm(real_dict["e_radial_ecal_r"], real_dict["e_radial_hcal_r"])
    plt_radial_mm(real_rad_mm, [fake_rad_mm], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2)
    print('done radial mm')
    sys.exit()
     
    cog_r = [real_dict['cog_x_r']-0.5, real_dict['cog_y_r'], real_dict['cog_z_r']-0.5]
    cog = [[fake_dict['cog_x']], [fake_dict['cog_y']], [fake_dict['cog_z']] ]
    plt_cog(cog_r, cog, num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2)
    print('done cog')
    
    xyz_r = [real_dict['X_r'], real_dict['Y_r'], real_dict['Z_r']]
    xyz = [[fake_dict['X']], [fake_dict['Y']], [fake_dict['Z']] ] 
    plt_xyz(xyz_r, xyz, num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2)
    print('xyz done')
    
    plt_hit_e(real_dict["hits_list_r"], [fake_dict["hits_list"]], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2, plt_config=plt_config, title=plt_config.title)
    print('done vis energy')
    plt_sum_e(real_dict["e_sum_list_r"], [fake_dict["e_sum_list"]], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2, plt_config=plt_config, title=plt_config.title)
    print('done en sum')
    
    real_ratio =  real_dict["e_sum_list_r"] / (real_dict['inc_en'].reshape(-1)*1000)
    fake_ratio =  fake_dict["e_sum_list"] / (fake_dict['inc_en'].reshape(-1)*1000)
    plt_e_ratio(real_ratio, [fake_ratio], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2, plt_config=plt_config, title=plt_config.title)
    print('done e ratio')
    
    plt_Nhits(real_dict["occ_list_r"], [fake_dict["occ_list"]], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2, plt_config=plt_config, title=plt_config.title)
    print('done Nhits')
    plt_startlayer(real_dict["start_layer_list_r"], [fake_dict["start_layer_list"]], num_of_shw, labels=['geant4', my_model_label], my_dir=directory_2, plt_config=plt_config, title=plt_config.title)
    print('done start layer')
    
    pearson_plot([real_dict['cog_x_r'], real_dict['cog_y_r'], real_dict['cog_z_r'], 
                  real_dict["occ_list_r"],real_dict["start_layer_list_r"], 
                  real_dict['inc_en'].reshape(-1)*1000, real_dict["e_sum_list_r"], 
                  real_dict["e_sum_ecal_list_r"], real_dict["e_sum_hcal_list_r"]],
                 [fake_dict['cog_x'], fake_dict['cog_y'], fake_dict['cog_z'], 
                  fake_dict["occ_list"],fake_dict["start_layer_list"],
                  fake_dict['inc_en'].reshape(-1)*1000, fake_dict["e_sum_list"], 
                  fake_dict["e_sum_ecal_list"], fake_dict["e_sum_hcal_list"]],
                 labels=['$cog_{x}$', '$cog_{y}$', '$cog_{z}$','$N_{hits}$', '$Y_{start}$', 
                         '$E_{inc}$', '$E_{sum}$', '$E_{sum}~ecal$','$E_{sum}~hcal$'], 
                 names= ['Geant4', my_model_label],
                 my_dir=directory_2)
    print('done pearson') 
    
    os.makedirs(directory_2+'WD_tables/', exist_ok=True)
    dir3 = directory_2+'WD_tables/'
    print('     Table AAL...') 
    kl_dictionary = kl_table(real_dict, fake_dict, shw=num_of_shw, threshold=threshold, my_dir = dir3)
    wd_dictionary = wd_table(real_dict, fake_dict, shw=num_of_shw, threshold=threshold, my_dir = dir3)
    plt_wdPlot(wd_dictionary, kl_dictionary, my_dir=dir3)
    
    print('\n   Table wdist...') 
    for key, value in wd_dictionary.items(): 
        print(key, ':  ', value)
        
    print('\n   Table kl...') 
    for key, value in kl_dictionary.items(): 
        print(key, ':  ', value)