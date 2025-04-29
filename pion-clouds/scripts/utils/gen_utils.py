import numpy as np
import torch
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import h5py
from scipy.stats import wasserstein_distance
from utils.dataset import PionClouds as pc
import time 

def get_only_hcal_data(events):
    # use only hcal and ecal part as conditioning
    dim_a, dim_b, dim_c = events.shape
    events2 = events.copy()
    condition_ecal = (events2[:, :, 1] < 30) & (events2[:, :, 3] > 0) 
    max_point_ecal = np.sum(condition_ecal, axis=1).max()
    max_point_hcal = np.sum(events2[:, :, 1] >= 30, axis=1).max()
    
    #re-do padding mask. Here is needed because the points are not ordered by n_points in the shower anymore!
    padding_mask_hcal = np.zeros((dim_a, max_point_hcal)).astype(bool)
    padding_mask_ecal = np.zeros((dim_a, max_point_ecal)).astype(bool) 
        
    events, cond_ecal = np.zeros((dim_a, max_point_hcal, dim_c)) , np.zeros((dim_a, max_point_ecal, dim_c))
    for j in range(events2.shape[0]):
        condition_ecal = (events2[j, :, 1] < 30) & (events2[j, :, 3] > 0)
        to_concatenate = max_point_ecal - np.sum(condition_ecal) 
        cond_ecal[j] = np.concatenate((events2[j, condition_ecal], np.zeros((to_concatenate, dim_c))), axis=0)
        idx_mask = np.where(cond_ecal[j,:,3]<=0)[0]
        padding_mask_ecal[j, idx_mask] = True 
        
        to_concatenate = max_point_hcal - np.sum(events2[j, :, 1] >= 30) 
        events[j] = np.concatenate((events2[j, events2[j, :, 1] >= 30], np.zeros((to_concatenate, dim_c))), axis=0)
        idx_mask = np.where(events[j,:,3]<=0)[0]
        padding_mask_hcal[j, idx_mask] = True   
    del events2 
    return events, cond_ecal, padding_mask_hcal, padding_mask_ecal 

def plot_not_proj(real_showers, fake_showers, cond_E_real, cond_E_fake, only_hcal=False, only_ecal=False, name='', sf=None):
    name0 = name.split('_')[0]
    r=[-470,470]
    if only_ecal: cond_real = (real_showers[:,3,:]>0) & (real_showers[:,1,:]<30)
    else: cond_real = real_showers[:,3,:]>0 
    aa,bb = 3, 5
    plt.figure(111, figsize=(70,30))
    plt.subplot(aa,bb ,1)
    b = 30
    x_fk = fake_showers[:,0,:][fake_showers[:,3,:]>0]
    plt.hist(real_showers[:,0,:][cond_real], bins=b, range = r, alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,0,:][fake_showers[:,3,:]>0], bins=b, range = r, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('x coordiante')
    plt.yscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,2)    
    if only_ecal: b1=np.linspace(0, 29, 30) 
    else: b1=78 
    # if fake_showers[:,1,:].max() < 1.5: 
    #     _range = [-1, 1]
    #     if only_hcal: b1=np.linspace(-1, 1, 50)
    #     # this is because I shift the range from (0,30) to (-1,1) but the maximum value is 29 so when I do from -1 to 1 I have to use 31 bins not 30! 
    #     # this corresponds to 32 edges
    #     elif only_ecal: b1=np.linspace(-1, 1, 32) 
    #     else: b1=np.linspace(-1, 1, 80)
    #     if sf is not None:
    #         to_add = np.zeros((sf.shape[0],1))
    #         sf = np.concatenate((sf, to_add), axis=1)
    # else:
    # shift = 0.5
    # if only_hcal: b1=np.linspace(30-shift, 78-shift, 49)
    # elif only_ecal: b1=np.linspace(0-shift, 30-shift, 31)  
    # else: b1=np.linspace(0-shift, 78-shift, 79) # 0.5, 78+1.5 
        
    # _range = [fake_showers[:,1,:][fake_showers[:,3,:]>0].min(), fake_showers[:,1,:][fake_showers[:,3,:]>0].max()]   
    # if sf is not None:
    #     if only_ecal: Ymax=30
    #     else: Ymax=78
    #     plt.stairs(sf.sum(axis=0) -Ymax, b1, linewidth=4, label='SF', color='red')
    ff = fake_showers[:,1,:][fake_showers[:,3,:]>0]
    rr = real_showers[:,1,:][cond_real]
    # if ff.max() > 10: 
    #     ff = ff.astype(int)
    #     rr = rr.astype(int) 
    plt.hist(rr, bins=b1, alpha=0.5, label='sim', color='grey')
    plt.hist(ff, bins=b1, linewidth=4, histtype='step', label='gen', color='green') #, 
    plt.xlabel('y coordiante')
    plt.yscale('log')
    plt.ylabel('counts')
    # plt.ylim([1e2, 5e4])
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,3)
    b=30
    z_fk = fake_showers[:,2,:][fake_showers[:,-1,:]>0]
    plt.hist(real_showers[:,2,:][cond_real], bins=b, range = r, alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,2,:][fake_showers[:,3,:]>0], bins=b, range = r, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('z coordinate')
    plt.yscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,4)
    b=70
    plt.hist(real_showers[:,3,:][cond_real].flatten(), bins = np.logspace(np.log10(1e-3), np.log10(1e3), b), alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,3,:].flatten() , bins= np.logspace(np.log10(1e-3), np.log10(1e3), b), linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('visible energy (log)')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
     
    plt.subplot(aa,bb,5)
    b=90 
    _range=(np.nanmin(fake_showers[:,3,:].sum(axis=1)), np.nanmax(fake_showers[:,3,:].sum(axis=1)))
    if only_ecal:
        real_sum, n_hits_real = [], []
        for j in range(real_showers.shape[0]):
            real_sum.append(real_showers[j,3,:][(real_showers[j,3,:]>0) & (real_showers[j,1,:]<30)].sum())
            n_hits_real.append(((real_showers[j,3,:]>0) & (real_showers[j,1,:]<30)).sum())
        real_sum = np.array(real_sum).flatten()
        n_hits_real = np.array(n_hits_real).flatten()
    else: 
        real_sum = (real_showers[:,3,:].sum(axis=1)).flatten()
        n_hits_real = ((real_showers[:,3,:]>0).sum(axis=1)).flatten()
        
    cond_e_r = cond_E_real.flatten() *1000
    cond_e_f = (cond_E_fake.flatten() +1)/2 *100 *1000
    plt.hist(real_sum / cond_e_r, bins = b, alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,3,:].sum(axis=1).flatten() /cond_e_f, bins= b, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('Energy Ratio')
    plt.yscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,6)
    if only_hcal: b1=np.linspace(30, 78, 49)
    elif only_ecal: b1=np.linspace(0-0.5, 30-0.5, 31)
    else: b1=np.linspace(0, 78, 79)
    if real_showers[:,1,:].max()<1.5:
        if only_hcal: b1=np.linspace(-1, 1, 50)
        elif only_ecal: b1=np.linspace(-1, 1, 32)
        else: b1=np.linspace(-1, 1, 80)
    # if name0=='': pos1 = pos1+0.5
    energy_per_layer_r, energy_per_layer  = [], []
    
    for layer in b1[:-1]: # from 0 to 77
        layer_mask = (real_showers[:,1,:]>=layer) & (real_showers[:,1,:]<(layer+1)) & (real_showers[:,3,:]>0)
        energy_per_layer_r.append(real_showers[:,3,:][layer_mask].sum())
        layer_mask = (fake_showers[:,1,:]>=layer) & (fake_showers[:,1,:]<(layer+1)) & (fake_showers[:,3,:]>0) 
        energy_per_layer.append(fake_showers[:,3,:][layer_mask].sum()) 
        
    energy_per_layer_r = np.array(energy_per_layer_r) /int(real_showers.shape[0])
    energy_per_layer = np.array(energy_per_layer) /int(fake_showers.shape[0])  
        
    # plt.step(np.array(energy_per_layer_r), b1, linewidth=4, alpha=0.5, label='sim', color='grey')
    # plt.step(np.array(energy_per_layer), b1, linewidth=4, label='gen', color='green')
    plt.xlabel('layers')
    plt.yscale('log')
    # plt.ylim([1e-1, 10])
    plt.ylabel('Mean energy')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,7)    
    _range = [0, np.nanmax(n_hits_real+100)] 
    plt.hist(n_hits_real, bins=60, range=_range, alpha=0.5, label='sim', color='grey')
    plt.hist((fake_showers[:,3,:]>0).sum(axis=1), bins=60, range=_range, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('# hits')
    plt.ylabel('counts (log scale)')
    plt.legend()
    plt.grid()
    
    sum_hcal_real, sum_hcal_fake = np.zeros(real_showers.shape[0]), np.zeros(real_showers.shape[0])
    sum_ecal_real, sum_ecal_fake = np.zeros(real_showers.shape[0]), np.zeros(real_showers.shape[0])
    for j in range(real_showers.shape[0]):
        if only_ecal==False:
            sum_hcal_real[j] = np.sum(real_showers[j,3,:][real_showers[j,1,:]>=30])
            sum_hcal_fake[j] = np.sum(fake_showers[j,3,:][fake_showers[j,1,:]>=30]) 
        sum_ecal_real[j] = np.sum(real_showers[j,3,:][real_showers[j,1,:]<30])
        sum_ecal_fake[j] = np.sum(fake_showers[j,3,:][fake_showers[j,1,:]<30])
    
    if only_ecal:
        plt.subplot(aa,bb,8)
        b=90
        cond_E_fake = (cond_E_fake +1) /2 *100
        _range=(np.nanmin(cond_E_fake), np.nanmax(cond_E_fake))
        plt.hist(cond_E_real.flatten(), bins = b, range = _range, alpha=0.5, label='sim', color='grey')
        plt.hist(cond_E_fake.flatten(), bins= b, range = _range, linewidth=4, histtype='step', label='gen', color='green')
        plt.xlabel('incident energy')
        plt.yscale('log')
        plt.ylabel('counts')
        plt.legend()
        plt.grid() 
    else:
        plt.subplot(aa,bb,8)
        b=90
        _range=(np.nanmin(sum_hcal_fake), np.nanmax(sum_hcal_fake))
        plt.hist(sum_hcal_real, bins = b, range = _range, alpha=0.5, label='sim', color='grey')
        plt.hist(sum_hcal_fake, bins= b, range = _range, linewidth=4, histtype='step', label='gen', color='green')
        plt.xlabel('energy sum HCAL')
        plt.yscale('log')
        plt.ylabel('counts')
        plt.legend()
        plt.grid()
    
    plt.subplot(aa,bb,9)
    b=90
    _range=(np.nanmin(sum_ecal_fake), np.nanmax(sum_ecal_fake))
    plt.hist(sum_ecal_real, bins = b, range = _range, alpha=0.5, label='sim', color='grey')
    plt.hist(sum_ecal_fake, bins= b, range = _range, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('energy sum ECAL')
    plt.yscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,10)
    b=90
    _range=(np.nanmin(fake_showers[:,3,:].sum(axis=1)), np.nanmax(fake_showers[:,3,:].sum(axis=1)))
    plt.hist(real_showers[:,3,:].sum(axis=1).flatten(), bins = b, range = _range, alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,3,:].sum(axis=1).flatten() , bins= b, range = _range, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('energy sum')
    plt.ylabel('counts (linear scale)')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,11)
    plt.hist(real_showers[:,3,:][(real_showers[:,1,:]<30) & (real_showers[:,3,:]>0)].flatten(), bins = np.logspace(np.log10(1e-3), np.log10(1e3), b), alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,3,:][(fake_showers[:,1,:]<30) & (fake_showers[:,3,:]>0)].flatten() , bins=np.logspace(np.log10(1e-3), np.log10(1e3), b), linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('visible energy (log) ECAL')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,12)
    plt.hist(real_showers[:,3,:][(real_showers[:,1,:]>29) & (real_showers[:,3,:]>0)].flatten(), bins = np.logspace(np.log10(1e-3), np.log10(1e3), b), alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,3,:][(fake_showers[:,1,:]>29) & (fake_showers[:,3,:]>0)].flatten() , bins=np.logspace(np.log10(1e-3), np.log10(1e3), b), linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('visible energy (log) HCAL')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
     
    plt.savefig('gen_utils_plot/after_sample_new_'+name+'.png')
    plt.close()

def plotting_pionclouds(real_showers, fake_showers, only_hcal=False, only_ecal=False, name=''):
    name0 = name.split('_')[0]
    shw = real_showers.shape[0]
    all_index = np.array([int(shw/50),int(shw/40), int(shw/30), int(shw/20), int(shw/10), int(shw/8),int(shw/6), int(shw/5), int(shw/4), int(shw/3), int(shw/2),int(shw/1.8), int(shw/1.7),int(shw/1.5), int(shw/1.4),int(shw/1.37), int(shw/1.3), int(shw/1.25), int(shw/1.2), int(shw/1.15), int(shw/1.1), int(shw/1.08), int(shw/1.05),int(shw/1.03), int(shw)-1])
    # for index in all_index: print( real_showers[index,3][real_showers[index,3]>0].shape )
    thr = 0.05
    if name0!='':
        myfig = plt.figure(10, figsize=(30,15))
        for i in range(all_index.shape[0]): 
            inp = real_showers[all_index[i]]
            ax = myfig.add_subplot(5, 5, i+1) #xy
            ax.scatter(inp[0][inp[3]>thr], inp[1][inp[3]>thr], s=inp[3][inp[3]>thr])
            if only_hcal: ax.set_ylim([30,78])
            elif only_ecal: 
                ax.set_ylim([0,30])
                if name0!='': ax.set_xlim([-400, 400])
            else: 
                ax.set_ylim([0,78])
                for y_line in [30]:
                    try:
                        ax.plot([inp[0][inp[3]>thr].min(), inp[0][inp[3]>thr].max()], [y_line,y_line], '-r')
                    except ValueError:  #raised if `y` is empty.
                        pass
            ax.axis('off') 
        myfig.savefig('gen_utils_plot/after_sample_clouds_real_'+name+'.png')
        plt.close()
    
    myfig2 = plt.figure(10, figsize=(30,15))
    for i in range(all_index.shape[0]):
        inp = fake_showers[all_index[i]]     
        ax2 = myfig2.add_subplot(5, 5, i+1) #xy
        ax2.scatter(inp[0][inp[3]>thr], inp[1][inp[3]>thr], s=inp[3][inp[3]>thr])
        if only_hcal: ax.set_ylim([30,78])
        elif only_ecal: 
            ax2.set_ylim([0,30])
            ax2.set_xlim([real_showers[all_index[i],0].min(), real_showers[all_index[i],0].max()])
        else:
            ax2.set_ylim([0,78])
            ax2.set_xlim([real_showers[all_index[i],0].min(), real_showers[all_index[i],0].max()])
            for y_line in [30]:
                try:
                    ax2.plot([inp[0][inp[3]>thr].min(), inp[0][inp[3]>thr].max()], [y_line, y_line], '-r')
                except ValueError:  #raised if `y` is empty.
                    pass
        ax2.axis('off')
    myfig2.savefig('gen_utils_plot/after_sample_clouds_fake_'+name+'.png')
    plt.close()
    
def _calculate_points_per_layer(events, axis=1, bins=78, only_hcal=False, only_ecal=False):
    points_per_layer = []
    # if events is standardize range=(-1,1) there is something wrong with the computation
    # keep events from 0 to 78 and it works best
    for event in events:
        num_p, _ = np.histogram(event[:, axis][event[:, 3] > 0], bins=bins, range=(0, bins)) 
        if only_hcal: num_p = num_p[30:]
        if only_ecal: num_p = num_p[:30]   
        points_per_layer.append(num_p)
    return np.array(points_per_layer)

def get_shower(model, num_points, energy, cond_Nperlayer=None, theta=0, phi=0, bs=1, cond_N=None, pm=None, cond_ecal = None, config=None, clock_dict=None):
    
    e = torch.ones((bs, 1), device=config.device) * energy
    p = torch.Tensor(cond_Nperlayer).to(config.device)

    padding_mask , cond_ecal_feats = None, None 
    if pm is not None: padding_mask = torch.Tensor(pm).to(config.device)
    if cond_ecal is not None: cond_ecal_feats = torch.Tensor(cond_ecal).to(config.device)
    
    cond_feats = torch.cat([e, p], -1)
    
    with torch.no_grad():
        padding_mask = None # the padding mask is not needed when sampling from the diffusion model 
        fake_shower = model.sample(cond_feats, num_points, config, padding=padding_mask, cond_ecal=cond_ecal_feats)
    return fake_shower

def x_z_shift(showers, Xmin, Xmax, Zmin, Zmax):
    showers = np.moveaxis(showers, -1, -2)   # (bs, num_points, 4) -> (bs, 4, num_points)
    showers[:, 0, :] = (showers[:, 0, :] + 1) / 2  
    showers[:, 2, :] = (showers[:, 2, :] + 1) / 2 
    
    showers[:, 0, :] = showers[:, 0, :] * (Xmax - Xmin) + Xmin
    showers[:, 2, :] = showers[:, 2, :] * (Zmax - Zmin) + Zmin
    return showers
    
def plot_SF(points_per_layer, clusters_per_layer, name='', outpath=''):
    _rangeECAL = (np.nanmin([points_per_layer[:, :30].flatten(), clusters_per_layer[:, :30].flatten()]), 
                  np.nanmax([points_per_layer[:, :30].flatten(), clusters_per_layer[:, :30].flatten()])
                  )
    _rangeHCAL = (np.nanmin([points_per_layer[:, 30:].flatten(), clusters_per_layer[:, 30:].flatten()]), 
                  np.nanmax([points_per_layer[:, 30:].flatten(), clusters_per_layer[:, 30:].flatten()])
                  )
    
    plt.figure(2, figsize=(35, 15))
    for layer in tqdm(range(78)):
        _range = _rangeECAL if layer < 30 else _rangeHCAL
        plt.subplot(6, 13, layer+1)
        plt.hist(points_per_layer[:, layer], bins=40, range=_range, alpha=0.9, color='gray')
        plt.hist(clusters_per_layer[:, layer], bins=40, range=_range, histtype='step', color='g')
        plt.xlabel('Points in Layer '+str(layer), fontsize=12)
        plt.yscale('log')
    plt.tight_layout()
    plt.savefig(outpath + 'distribution_'+name+'.png')
    plt.close() 
                    
    plt.figure(2, figsize=(8, 8))
    _range = (0, 3100)
    plt.hist(points_per_layer.sum(axis=1), bins=40, range=_range, alpha=0.9, color='gray', label='geant4')
    plt.hist(clusters_per_layer.sum(axis=1), bins=40, range=_range, histtype='step', color='g', linewidth=3, label='sum')
    plt.xlabel('Total Points ', fontsize=12)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath + 'hits_tot_'+name+'.png')
    plt.close()    
    
    # error every 5k
    tot = points_per_layer.shape[0] 
    wdist_tot, wdist_per_layer_list, std_per_layer_list = [], [], []
    for j in tqdm(np.arange(0, tot, 5000), mininterval=3):
        wdist_tot.append(wasserstein_distance(points_per_layer.sum(axis=1)[j:j+1], clusters_per_layer.sum(axis=1)[j:j+1]))
        wdist_per_layer = []
        for l in range(78):
            wdist_per_layer.append(wasserstein_distance(points_per_layer[j:j+1, l], clusters_per_layer[j:j+1, l]))
        wdist_per_layer_list.append(np.mean(wdist_per_layer))
          
    wdist_tot = np.array(wdist_tot) 
    wdist_per_layer = np.array(wdist_per_layer_list)  
    
    with open(outpath+name+'.txt', 'w') as file:
        file.write('Wasserstein distance tot: '+str(np.mean(wdist_tot))+ ' +- '+str(np.std(wdist_tot))+'\n')
        file.write('Wasserstein distance per layer: '+str(np.mean(wdist_per_layer_list))+ ' +- '+str(np.std(wdist_per_layer_list))+'\n')
      
def gen_showers_batch(model, shower_flow, e_min, e_max, theta=0, phi=0, num=2000, bs=8, kdiffusion=False, 
                      config=None, max_points=5131, enable_shower_flow=True, cond_E=None, cond_ECAL=None,
                      single_SF=None, coef_real=None, coef_fake=None, n_scaling = False, n_splines=None,
                      gen_both_EandHcal=False, padding_mask_ecal=None, real_showers=None, cond_E_real=None, clock_dict =None):
    # name samples #hgx9 --> max points = 4048
     
    Xmax, Xmin, Zmax, Zmin, Ymin, Ymax = 450, -450, 450, -450, 0 ,78 
    if config.only_hcal: Ymin = 30 
    elif config.only_ecal: Ymax = 30 

    if real_showers is None: 
        f = h5py.File(config.data.dataset_path.format(30), 'r')
        events = np.array(f["events"][:])
        energy = np.array(f["energy"][:])
        if config.only_ecal: n_points = np.array(f['n_points_ecal'][:]) 
        if config.only_hcal: n_points = np.array(f['n_points_hcal'][:])
        else: n_points = np.array(f['n_points'][:]) 
        energy_mask = (energy.reshape(-1) > e_min) & (energy.reshape(-1) < e_max)
        idx = np.sort((np.random.rand(1,num) * energy_mask.sum())).astype(int)[0]
        idx_sorted = np.argsort(n_points[energy_mask][idx])
        n_points = n_points[energy_mask][idx][idx_sorted]
        real_showers = events[energy_mask][idx][idx_sorted][:, [0,1,2,3]]
        cond_E_real = torch.Tensor(energy[energy_mask][idx][idx_sorted]).to(config.device)
        max_len = (real_showers[:, 3] > 0).sum(axis=1).max()
        real_showers = real_showers[:, :, -max_len:]
        real_showers[:, 3, :] *= 1000   
        
    if config.only_hcal: 
        samples = _calculate_points_per_layer(np.moveaxis(real_showers,-1,-2)) 
        real_showers = np.moveaxis(real_showers, -1, -2)
        real_showers[real_showers[:, :, 3] == 0] = 0 
        real_showers2, tot_cond_ecal, _, _ = get_only_hcal_data(real_showers) 
        real_showers2 = np.moveaxis(real_showers2, -1, -2)   
        cond_E = cond_E_real / 100 * 2 -1 
        tot_cond_ecal[:, :, 0] = (tot_cond_ecal[:, :, 0] - Xmin) * 2 / (Xmax - Xmin) - 1  # x normalization
        tot_cond_ecal[:, :, 2] = (tot_cond_ecal[:, :, 2] - Zmin) * 2 / (Zmax - Zmin) - 1  # z normalization  
         
    if config.only_ecal:
        condition_ecal = (real_showers[:, 1] < 30) & (real_showers[:, 3] > 0) 
        max_point_ecal = np.sum(condition_ecal, axis=1).max()
        max_len = (real_showers[:, 3] > 0).sum(axis=1).max()
        real_showers = real_showers[:, :, -max_len:]
        real_showers = np.moveaxis(real_showers, -1, -2)
        real_showers[real_showers[:, :, 3] == 0, :] = 0 
        real_showers2, tot_cond_ecal, tot_padding_mask, tot_padding_mask_ecal = get_only_hcal_data(real_showers) 
        real_showers = np.moveaxis(real_showers, -1, -2)
        real_showers2 = np.moveaxis(real_showers2, -1, -2)
        tot_padding_mask = tot_padding_mask_ecal
        e = cond_E_real.detach().cpu().numpy()
                           
    if config.only_ecal:
        cond_E = cond_E.to(config.device)
        # the incident energy for the SF is initialized just diving for the max
        en_max_sf = 90.1395492553711
        context = ((cond_E +1) /2 *100) / en_max_sf
        p_per_l_max = 6.400257445308821   #in log scale!
        # sample from shower flow  
        if (single_SF is not None and config.only_ecal) | (single_SF is None): 
            # samples = shower_flow.condition(context).sample(torch.Size([num, ])).cpu().numpy() # 126.7 max energy label in the dataset
            
            with torch.no_grad():
                samples = shower_flow.sample((context.shape[0], 78), condition=context).cpu().numpy()
            
            samples = np.exp(samples * p_per_l_max).astype(int) 
            
            # this is a control on the number of points per layer, and the total number of points
            p=0
            while (len(np.where(samples.sum(axis=1)>5001)[0])>0) | (len(np.where(samples>602)[0])>0):
                indexes = np.unique(np.concatenate([np.where(samples.sum(axis=1)>5001)[0], np.where(samples>602)[0]]))
                for i in indexes:
                    with torch.no_grad():
                        samples[i] = shower_flow.sample((1, 78), condition=context[i].reshape(1,1)).cpu().numpy()
                        samples[i] = np.exp(samples[i] * p_per_l_max).astype(int)
                    p+=1
                    if p>10: break
            samples = samples.astype(int)        
        else: samples = single_SF[:,:78].copy() 
    else:
        samples = single_SF.copy() 

    mmaxx = samples.sum(axis=1).max()
    if config.only_hcal: hits_per_layer_all_tot = samples[:, 30:]
    elif config.only_ecal: hits_per_layer_all_tot = samples[:, :30]      
    else: hits_per_layer_all_tot = samples 
           
    # ordering per number of hits 
    idx_hits_sorted = np.argsort(hits_per_layer_all_tot.sum(axis=1)) 
    hits_per_layer_all_tot = hits_per_layer_all_tot[idx_hits_sorted]
    cond_E = cond_E[idx_hits_sorted]
    samples = samples[idx_hits_sorted]
    hits_per_layer_all_tot2 = samples    
    if config.only_hcal: 
        tot_cond_ecal = tot_cond_ecal[idx_hits_sorted]
        cond_E = torch.Tensor(cond_E).to(config.device)
        points_ECAL = torch.Tensor(np.sum(tot_cond_ecal[:,:,3]>0, axis=1)).reshape(num, 1).to(config.device)       
    if (enable_shower_flow==False) & (config.only_ecal): real_showers2 = real_showers2[idx_hits_sorted]     
    
    fake_showers_list, fs2_list, real_list, real_list2 = [],[], [], []
    fs3_list, real_list3 =[], [] 
    cond_ecal_tosave_list, samples_list = [], [] 
    padding_mask_list = []                       
    shw = num 
    # all_index = np.array([int(shw/50),int(shw/40), int(shw/30), int(shw/20), int(shw/10), int(shw/8),int(shw/6), int(shw/5), int(shw/4), int(shw/3), int(shw/2),int(shw/1.8), int(shw/1.7),int(shw/1.5), int(shw/1.4),int(shw/1.37), int(shw/1.3), int(shw/1.25), int(shw/1.2), int(shw/1.15), int(shw/1.1), int(shw/1.08), int(shw/1.05),int(shw/1.03), int(shw)-1])
    all_index = np.array([int(shw/1.15), int(shw/1.1), int(shw/1.08), int(shw/1.05),int(shw/1.03), int(shw)-1])
    counts_rejected = 0 
    max_shower_length = mmaxx + 10
    
    for evt_id in tqdm(range(0, num, bs), mininterval = 10, disable=False):
        if (num - evt_id) < bs: 
            bs = num - evt_id 
            
        cond_E_batch = cond_E[evt_id : evt_id+bs]    
        hits_per_layer_all = hits_per_layer_all_tot[evt_id : evt_id+bs]     
        num_clusters = np.sum(hits_per_layer_all, axis=1).reshape(bs, 1) #B,1 
        if len(np.where(hits_per_layer_all<0)[0])!=0: hits_per_layer_all[hits_per_layer_all<0]=0
                
        cond_N = torch.Tensor(hits_per_layer_all.sum(axis=1) / max_points).to(config.device).unsqueeze(-1)
        max_num_clusters = hits_per_layer_all.sum(axis=1).max()
        padding_mask = np.zeros((bs, max_num_clusters)).astype(bool)
        col_indices = np.arange(max_num_clusters)
        padding_mask[col_indices >= np.array(num_clusters.reshape(-1))[:, None]] = True 
        cond_Nperlayer = torch.Tensor(hits_per_layer_all/ hits_per_layer_all.max()).to(config.device)  
        
        if config.only_hcal: 
            cond_ecal = tot_cond_ecal[evt_id : evt_id+bs]  
            cond_ecal_temp = tot_cond_ecal[evt_id : evt_id+bs]  
            # preprocessing ecal data  
            _, cond_ecal, _, padding_mask_ecal = get_only_hcal_data(cond_ecal)
            cond_ecal_tosave = cond_ecal.copy()
            smearing = np.random.uniform(-0.49, 0.49, size=cond_ecal[:, :, 1].shape)
            cond_ecal[:, :, 1] = cond_ecal[:, :, 1] + smearing
            cond_ecal[:, :, 1] = (cond_ecal[:, :, 1] - 0) * 2 / (30 - 0) - 1 # y normalization 
            max_point_ecal = points_ECAL[evt_id : evt_id+bs].max().reshape(-1)
            cond_ecal = torch.Tensor(cond_ecal).to(config.device)      
        else: cond_ecal, cond_ecal_temp, cond_ecal_tosave = None, None, None 
        
        #generationnnnnnn 
        fs = get_shower(model, max_num_clusters, cond_E_batch, cond_N = cond_N, cond_Nperlayer = cond_Nperlayer, cond_ecal = cond_ecal, 
                            theta=0, bs=bs, pm = padding_mask, config=config, clock_dict=clock_dict_)
        
        if config.data.log_energy: 
            fs[:, :, 3] *= config.data.log_var
            fs[:, :, 3] += config.data.log_mean
            fs[:, :, 3] = torch.exp(fs[:, :, 3])     
        fs = fs.cpu().numpy()
        
        if config.only_ecal: max_clip = 29 /30 *2-1 # layer 29 is the max
        elif config.only_hcal: max_clip = (77-30) /48 *2-1 # layer 77 is the max
        
        if config.only_hcal: Ymin = 30        
        fs[:, :, 1] = np.clip(fs[:, :, 1], -1, max_clip)
        fs[:, :, 1] = (fs[:, :, 1] + 1) / 2 
        fs[:, :, 1] = fs[:, :, 1] * (Ymax - Ymin) + Ymin 
        for i in range(4): fs[:,:,i][padding_mask] = 0        
        fs2 = fs.copy()
        
        if config.only_ecal: 
            for i in range(4): fs[:,:,i][padding_mask] = -10 # for SF calibration  
        if config.only_hcal: 
            cond_ecal = cond_ecal_tosave 
            padding_mask = np.concatenate((padding_mask_ecal, padding_mask), axis=1).astype(bool)
            fs2 = np.concatenate((cond_ecal, fs), axis=1).copy()
            fs = np.concatenate((cond_ecal, fs), axis=1)
            max_num_clusters = int(max_num_clusters + max_point_ecal) 
            for i in range(4): fs[:,:,i][padding_mask] = -10 # for SF calibration  

        if config.only_hcal: #config.only_hcal: 
            Ymin, Ymax = 0, 78
            hits_per_layer_all = hits_per_layer_all_tot2[evt_id : evt_id+bs] #[:, :Ymax]
        elif config.only_ecal:
            Ymin, Ymax = 0, 30
            hits_per_layer_all = hits_per_layer_all_tot2[evt_id : evt_id+bs][:, :30] #[:, :Ymax]
        
        cell_thickness = 1
        layer_bottom_pos = np.linspace(Ymin, Ymax-1, Ymax-Ymin) 
        y_positions = layer_bottom_pos+cell_thickness/2 
        
        for i, hits_per_layer in enumerate(hits_per_layer_all): 
            hits_per_layer[hits_per_layer<0] = 0
            n_hits_to_concat = max_num_clusters - hits_per_layer.sum() 
            z_flow = np.repeat(y_positions, hits_per_layer)           
            
            if hits_per_layer.sum() > max_num_clusters: 
                print('something is wrong! hits per layer > max num clusters') 
                print(hits_per_layer.sum(), max_num_clusters)
                n_hits_to_concat = 0
                z_flow = z_flow[:max_num_clusters]   
            
            z_flow = np.concatenate([z_flow, np.zeros(n_hits_to_concat)])
            if padding_mask is not None:
                mask = padding_mask[i,:] # note!--> this is already inverted because of attention
                fs[i, :, 1][mask] = 100
            else: 
                mask = np.concatenate([np.zeros(n_hits_to_concat), np.ones(hits_per_layer.sum())])  
                fs[i, :, 1][mask == 0] = 100 

            idx_dm = np.argsort(fs[i, :, 1]) 
            fs[i, :, :] = fs[i, :, :][idx_dm] 
            fs[i, :, 1] = z_flow    
            
            if (config.only_hcal==False) & (config.only_ecal==False): z_flow = np.sort(z_flow)      
            if fs[i, :, :].shape[0] != z_flow.shape:
                z_flow = z_flow[:fs[i, :, :].shape[0]]
            else:  
                for f in range(4): fs[i, :, f][z_flow == 0] = 0
            fs[fs[:, :, 3]  <= 0] = 0    # setting events with negative energy to zero          
                  
        length = max_shower_length - fs.shape[1] 
        if length < 0:
            print('something is wrong! length < 0')
            print(max_shower_length, fs.shape[1])
            print(mmaxx)
            print('cutting the showers at length: ', max_shower_length)
            fs = fs[:, :max_shower_length, :]
            fs2 = fs2[:, :max_shower_length, :]
            length = max_shower_length - fs.shape[1]
        else:
            fs = np.concatenate((fs, np.zeros((bs, length, 4))), axis=1)  # B, max_points, 4
            fs2 = np.concatenate((fs2, np.zeros((bs, length, 4))), axis=1)
         
        Xmax, Xmin, Zmax, Zmin = 450, -450, 450, -450 
        
        fs = x_z_shift(fs, Xmin, Xmax, Zmin, Zmax)
        fs2 = x_z_shift(fs2, Xmin, Xmax, Zmin, Zmax)   
        fs2_list.append(fs2)
        fake_showers_list.append(fs)
      
    fake_showers = np.vstack(fake_showers_list)            
    fake_showers2 = np.vstack(fs2_list)       
    
    fake_showers[:,1,:] = fake_showers[:,1,:] - 0.5
    fake_showers2[:,1,:] = fake_showers2[:,1,:].astype(int)
    
    flag=False 
       
    cond_E_fake = cond_E.detach().cpu().numpy()
    
    if config.only_hcal: nn='H'
    elif config.only_ecal: 
        nn='E'
        cond_E_real = cond_E_real.detach().cpu().numpy() 
                    
    plot_not_proj(real_showers, fake_showers2, cond_E_real, cond_E_fake, name='_'+nn, only_hcal=flag, only_ecal=config.only_ecal, sf = hits_per_layer_all_tot)  
    plotting_pionclouds(real_showers, fake_showers2, name='_'+nn, only_hcal=flag, only_ecal=config.only_ecal)
    
    if config.only_hcal:
        plot_not_proj(real_showers, fake_showers, cond_E_real, cond_E_fake, name='layerPP_'+nn, only_hcal=flag, only_ecal=config.only_ecal, sf = hits_per_layer_all_tot)
        plotting_pionclouds(real_showers, fake_showers, name= 'layerPP_'+nn, only_hcal=flag, only_ecal=config.only_ecal)     

    return fake_showers, samples, cond_E.detach().cpu().numpy().astype('float32'), real_showers, cond_E_real  

