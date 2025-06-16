from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import utils.metrics as metrics
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm 
from memory_profiler import profile
import sys 

mpl.rcParams['xtick.labelsize'] = 18    
mpl.rcParams['ytick.labelsize'] = 18
# mpl.rcParams['font.size'] = 28
mpl.rcParams['font.size'] = 25
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'serif'
# mpl.rcParams['axes.aspect'] = 'equal' 
# mpl.rcParams['figure.autolayout'] = True # for the hit e distr.
# mpl.rcParams['font.weight'] = 1  # Or you can use a numerical value, e.g., 700
plt.close()
font=25


def get_cog2(events, Xmin=-450, Xmax=450, cell_size = 1):
    n, Z, Y, X = events.shape
    x_vals = np.linspace(Xmin + cell_size/2, Xmax - cell_size/2, X)
    y_vals = np.linspace(0 + cell_size/2, 78 - cell_size/2, 78)
    z_vals = np.linspace(Xmin + cell_size/2, Xmax - cell_size/2, Z)
    zz, yy, xx = np.meshgrid(z_vals, y_vals, x_vals, indexing='ij')    
    xx = xx[None, :, :, :]  # shape: (1, Z, Y, X)
    yy = yy[None, :, :, :]
    zz = zz[None, :, :, :]
    
    total_energy = events.sum(axis=(1, 2, 3), keepdims=True)

    # Compute weighted sums
    cog_x = (events * xx).sum(axis=(1, 2, 3)) / total_energy.squeeze()
    cog_y = (events * yy).sum(axis=(1, 2, 3)) / total_energy.squeeze()
    cog_z = (events * zz).sum(axis=(1, 2, 3)) / total_energy.squeeze()
    
    cog = np.stack([cog_x, cog_y, cog_z], axis=1)
    return cog_x, cog_y, cog_z

def get_cog(events, Xmin=-450, Xmax=450, cell_size = 5): 
    y_values = np.arange(0, 78, 1)
    xz_values = np.arange(Xmin, Xmax, cell_size) # it is right like this
    
    sum_xz = events.sum(axis=3).sum(axis=1)
    cog_y = np.sum(sum_xz * y_values, axis=1) / sum_xz.sum(axis=1)
    sum_yz = events.sum(axis=3).sum(axis=2)
    cog_x = np.sum(sum_yz * xz_values, axis=1) / sum_yz.sum(axis=1)
    sum_yx = events.sum(axis=1).sum(axis=1)
    cog_z = np.sum(sum_yx * xz_values, axis=1) / sum_yx.sum(axis=1)
    return cog_x, cog_y, cog_z 

def plotting_pionclouds(directory, real_showers, fake_showers, only_hcal=False):
    shw = real_showers.shape[0]
    all_index = np.array([int(shw/50),int(shw/40), int(shw/30), int(shw/20), int(shw/10), int(shw/8),int(shw/6), int(shw/5), int(shw/4), int(shw/3), int(shw/2),int(shw/1.8), int(shw/1.7),int(shw/1.5), int(shw/1.4),int(shw/1.37), int(shw/1.3), int(shw/1.25), int(shw/1.2), int(shw/1.15), int(shw/1.1), int(shw/1.08), int(shw/1.05),int(shw/1.03), int(shw)-1])
    # for index in all_index: print( real_showers[index,3][real_showers[index,3]>0].shape )
    thr = 1e-1
    myfig = plt.figure(10, figsize=(30,15))
    for i in range(all_index.shape[0]): 
        inp = real_showers[all_index[i]]
        ax = myfig.add_subplot(5, 5, i+1) #xy
        ax.plot(inp[0][inp[3]>thr], inp[1][inp[3]>thr],'.')
        if only_hcal: ax.set_ylim([30,78])
        else: 
            ax.set_ylim([0,78])
            for y_line in [30]:
                ax.plot([inp[0][inp[3]>thr].min(), inp[0][inp[3]>thr].max()], [y_line,y_line], '-r')
        ax.axis('off') 
    myfig.savefig(directory+'after_sample_clouds_real.png')
    plt.close()
    
    myfig2 = plt.figure(10, figsize=(30,15))
    for i in range(all_index.shape[0]):
        inp = fake_showers[all_index[i]]     
        ax2 = myfig2.add_subplot(5, 5, i+1) #xy
        ax2.plot(inp[0][inp[3]>thr], inp[1][inp[3]>thr],'.')
        if only_hcal: ax.set_ylim([30,78])
        else:
            ax2.set_ylim([0,78])
            for y_line in [30]:
                ax2.plot([inp[0][inp[3]>thr].min(), inp[0][inp[3]>thr].max()], [y_line,y_line], '-r')
        ax2.axis('off')
    myfig2.savefig(directory+'after_sample_clouds_fake.png')
    plt.close()
    
def plot_not_proj( my_dir, real_showers, fake_showers, only_hcal_flag=False):
    x_z_lim = 450
    plt.figure(111, figsize=(90,40))
    aa,bb = 2, 4
    plt.subplot(aa,bb,1)
    b = 29
    r = [-x_z_lim-50, x_z_lim+50]  
    plt.hist(real_showers[:,0,:][real_showers[:,-1,:]>0], bins=b, range = r, alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,0,:][fake_showers[:,-1,:]>0], bins=b, range = r, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('x coordiante')
    plt.yscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,2)
    b2=np.arange(79) # 0.5, 78+1.5 # b2=np.concatenate((np.arange(1, 40), np.linspace(40, 80, 20)))
    if only_hcal_flag: b2=np.arange(30, 79) 
    plt.hist(real_showers[:,1,:][real_showers[:,3,:]>0], bins=b2, alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,1,:][fake_showers[:,3,:]>0], bins=b2, linewidth=4, histtype='step', label='gen', color='green') #, 
    plt.xlabel('y coordiante')
    plt.yscale('log')
    plt.ylabel('counts')
    # plt.ylim([1e3, 1e6])
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,3)
    b=49
    r = [-40,40] 
    cog_z_r = np.sum((real_showers[:,2] * real_showers[:,3]), axis=1) / real_showers[:,3].sum(axis=1)
    cog_z = np.sum((fake_showers[:,2] * fake_showers[:,3]), axis=1) / fake_showers[:,3].sum(axis=1)
    plt.hist(cog_z_r, bins=b, range=r, alpha=0.5, label='sim', color='grey')
    plt.hist(cog_z, bins=b, range=r, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('cog z coordinate')
    # plt.yscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,4)
    maxx = np.nanmax(fake_showers[:,-1,:].flatten())
    b=90
    plt.hist(real_showers[:,-1,:].flatten(), bins = np.logspace(np.log10(1e-3), np.log10(1e3), b), alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,-1,:].flatten() , bins=np.logspace(np.log10(1e-3), np.log10(1e3), b), linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('visible energy (log)')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,5)
    b=90
    _range=(np.nanmin(real_showers[:,3,:].sum(axis=1)), np.nanmax(real_showers[:,3,:].sum(axis=1)))
    plt.hist(real_showers[:,3,:].sum(axis=1).flatten(), bins = b, range = _range, alpha=0.5, label='sim', color='grey')
    plt.hist(fake_showers[:,3,:].sum(axis=1).flatten() , bins= b, range = _range, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('energy sum')
    plt.yscale('log')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,6)
    if only_hcal_flag: b1=np.linspace(30, 78, 49)
    else: b1=np.linspace(0, 78, 79)
    # pos1 = pos1+0.5 
    energy_per_layer_r, energy_per_layer  = [], []
    for layer in b1[:-1]: # from 0 to 77
        layer_mask = (real_showers[:,1,:]>layer) & (real_showers[:,1,:]<=(layer+1))
        energy_per_layer_r.append(real_showers[:,3,:][layer_mask].mean())
        layer_mask = (fake_showers[:,1,:]>layer) & (fake_showers[:,1,:]<=(layer+1))
        energy_per_layer.append(fake_showers[:,3,:][layer_mask].mean())
    
    # bar and step put the bins in different parts of the x axis     
    plt.stairs(np.array(energy_per_layer_r), b1, linewidth=1, alpha=0.5, label='sim', color='grey')
    plt.stairs(np.array(energy_per_layer), b1, linewidth=4, label='gen', color='green')
    plt.xlabel('layers')
    plt.yscale('log')
    plt.ylabel('Mean energy')
    plt.legend()
    plt.grid()
    
    plt.subplot(aa,bb,7)  
    _range = [np.nanmin((real_showers[:,3,:]>0).sum(axis=1)), np.nanmax((real_showers[:,3,:]>0).sum(axis=1))]   
    plt.hist((real_showers[:,3,:]>0).sum(axis=1), bins=60, range=_range, alpha=0.5, label='sim', color='grey')
    plt.hist((fake_showers[:,3,:]>0).sum(axis=1), bins=60, range=_range, linewidth=4, histtype='step', label='gen', color='green')
    plt.xlabel('# hits')
    plt.ylabel('counts')
    plt.legend()
    plt.grid()
    plt.savefig(my_dir+'new_not_proj.png')
  
def plotting_overlay2(my_dir, events, real=True, n_shw=1):
    if real: nn = 'real'
    else: nn = 'fake'
    all_index = np.array([int(shw/50),int(shw/40), int(shw/30), int(shw/20), int(shw/10), int(shw/8),int(shw/6), int(shw/5), int(shw/4), int(shw/3), int(shw/2),int(shw/1.8), int(shw/1.7),int(shw/1.5), int(shw/1.4),int(shw/1.37), int(shw/1.3), int(shw/1.25), int(shw/1.2), int(shw/1.15), int(shw/1.1), int(shw/1.08), int(shw/1.05),int(shw/1.03), int(shw)-1])
    # index = np.array([25]).astype(int)
    # index = np.sort((np.random.rand(1,n_shw) * shw)).astype(int)[0]
    thr = 0.15
    myfig = plt.figure(10, figsize=(40,30))
    for i in range(all_index.shape[0]):
        index = np.array([all_index[i]]).astype(int) 
        inp = np.array(events)[index]
        myfig.add_subplot(5,5,i+1) #xy
        to_plot_xy = np.moveaxis(np.sum(np.sum(inp, axis=0),axis=2),0,1)
        to_plot_xy[to_plot_xy <= thr] = None
        plt.imshow(to_plot_xy)
        if (name.split('_')[0]=='HGx9') | (name.split('_')[0]=='HGx4'): 
            plt.plot([50, 120], [30, 30], '-r')
            # plt.xlim([60, 120])
        else: plt.plot([-20, 60], [30, 30], '-r')
        plt.axis('off')
    if thr ==0: plt.savefig(my_dir+'/Overlay_prova1_'+nn+'_'+str(i)+'_'+str(shw)+ '_showers.png')
    else: plt.savefig(my_dir+'/Overlay_prova1_'+nn+'_'+str(i)+'_thr_'+str(thr)+'_MeV_'+str(shw)+ '_showers.png') 
    plt.close()
 
def plotting_overlay3(my_dir, events, real=True, shw=1):
    if real: nn = 'real'
    else: nn = 'fake'
    thr = 0.1 # MIP cut
    shw_per_plot = 10
    colums = int(shw_per_plot/2)
    rows = int(shw_per_plot/colums) 
    # here you plot 25 showers per loop
    kloop = np.arange(0, shw, shw_per_plot)
    for k in kloop:
        myfig = plt.figure(10, figsize=(50,20))
        for i in range(shw_per_plot):
            inp = np.array(events)[k+i]
            to_plot_xy = np.moveaxis(np.sum(inp, axis=2), 0, 1)
            to_plot_xy[to_plot_xy <= thr] = None
            ax = myfig.add_subplot(rows, colums, i+1) #xy
            ax.imshow(to_plot_xy)
            ax.plot([50,130], [30,30],'-r')
            ax.plot([90, 90], [0,78], '--b')
            ax.set_xlim([30,150])
            ax.axis('off')
        plt.savefig(my_dir+'/Overlay_num_'+str(int(k/shw_per_plot))+'_'+nn+'_thr_'+str(thr)+'_MeV_'+str(shw)+ '_showers.png') 
        plt.close()
        
def plotting_correlations(my_dir, events, events_r, axis=0): # axis= 0,1,3
    _range= [0, 1900] 
    cmin = 1 
    if axis==0: # number of showers axis 
        to_plot = np.array(events).sum(axis=3).sum(axis=1)     #sum over z and x 
        to_plot_r = np.array(events_r).sum(axis=3).sum(axis=1) #sum over z and x
    elif axis==1: # x axis 
        to_plot = np.array(events).sum(axis=3).sum(axis=0)     #sum over z and showers 
        to_plot_r = np.array(events_r).sum(axis=3).sum(axis=0) #sum over z and showers
    elif axis==3: # z axis
        to_plot = np.moveaxis(np.array(events).sum(axis=0).sum(axis=0), 1, 0)     # sum over showers and x
        to_plot_r = np.moveaxis(np.array(events_r).sum(axis=0).sum(axis=0), 1, 0) #sum over showers and x

    ecal = to_plot[:, :30].sum(axis=1) #sum over y ecal 
    hcal = to_plot[:, 30:].sum(axis=1) #sum over y hcal 
    ecal_r = to_plot_r[:, :30].sum(axis=1) #sum over y ecal 
    hcal_r = to_plot_r[:, 30:].sum(axis=1) #sum over y hcal 
    thebins = [50, 50]
    print('prepare for histograms')
    H, _, _ = np.histogram2d(ecal , hcal , bins=thebins, range=[_range, _range])
    H_r, _, _ = np.histogram2d(ecal_r , hcal_r , bins=thebins, range=[_range, _range])
    cmax = max(H.max(), H_r.max())
    print('done, now plotting... ')
    fig = plt.figure(10, figsize=(30,9))
    spec = gridspec.GridSpec(1, 4, width_ratios=[4, 4, 1.5, 4], wspace=0.3)
    # First subplot
    ax1 = fig.add_subplot(spec[0, 0])
    hist1 = ax1.hist2d(
        hcal, ecal, bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=mpl.colors.LogNorm()
    )
    ax1.set_title("CaloHadronic", fontsize=font)
    ax1.set_xlabel("energy hcal [MeV]", fontsize=font)
    ax1.set_ylabel("energy ecal [MeV]", fontsize=font)
    
    # Second subplot    
    ax2 = fig.add_subplot(spec[0, 1])
    hist2 = ax2.hist2d(
        hcal_r, ecal_r, bins=thebins, range=[_range, _range],
        cmin=cmin, cmax=cmax, norm=mpl.colors.LogNorm()
    )
    ax2.set_title("Geant4", fontsize=font)
    ax2.set_xlabel("energy hcal [MeV]", fontsize=font)
    ax2.set_ylabel("energy ecal [MeV]", fontsize=font)
    
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.axis('off')
    cbar1 = fig.colorbar(hist2[3], ax=ax3, orientation='vertical', pad=-0.1, location='left')
    cbar1.set_label("Counts", fontsize=font)
    # cbar1.ax.yaxis.set_ticks_position('left')
    # cbar1.ax.yaxis.set_label_position('left')
    
    # Third subplot    
    ax4 = fig.add_subplot(spec[0, 3]) 
    thr_ecal = 30
    plt.hist(hcal_r[ecal_r<thr_ecal], bins=30, color='gray', alpha=0.5, label='geant4', density=True) 
    plt.hist(hcal[ecal<thr_ecal], bins=30, color='purple', label='gen', histtype='step', linewidth=4, density=True) 
    plt.legend(fontsize=font)
    ax4.set_title(f"energy ecal < {thr_ecal:.0f} [MeV]", fontsize=font) 
    ax4.set_xlabel("energy hcal [MeV]", fontsize=font) 
    ax4.set_ylabel("Normalized", fontsize=font) 
    
    if axis==0: plt.savefig(f"{my_dir}/Correlation_histograms.png")
    elif axis==1: plt.savefig(f"{my_dir}/Correlation_histograms_x.png") 
    elif axis==3: plt.savefig(f"{my_dir}/Correlation_histograms_z.png")  
    plt.close()
    
def plotting_correlations_withCOGy(my_dir, fake, real, inc_en,
                                labels=['$E_{sum}$', '$cog_{y}$'], 
                                names=['Geant4', 'CaloHadronic']): # axis= 0,1,3
    _range0= [9.5, 11.6] #[0, 2400]
    _range1= [0, 78]
    color_cmap = 'Oranges'  
    vmin = 0 
    to_plot = fake[0]     #sum over z and x 
    to_plot_r = real[0] #sum over z and x
    cog_y = fake[1] 
    cog_y_r = real[1]
    
    # # check the logarithm dependence
    # title = ["Incident Energy [MeV]", "Energy Sum [MeV]"]
    # arrays = [inc_en, [to_plot_r, to_plot]]
    # n_bins_list = [10, 30, 50, 70, 90]
    # j=0
    # for ii in range(len(n_bins_list)):
    #     for k in range(2):
    #         a, b = arrays[k]
    #         logE = np.log(a[inc_en[0]>0]+1e-10)
    #         logE_gen = np.log(b[inc_en[1]>0]+1e-10)
            
    #         idx = np.argsort(logE)
    #         idx_gen = np.argsort(logE_gen)
    #         logE = logE[idx].flatten()
    #         logE_gen = logE_gen[idx_gen].flatten()
    #         new_cog = cog_y_r[inc_en[0]>0][idx].flatten()
    #         new_cog_gen = cog_y[inc_en[1]>0][idx_gen].flatten()

    #         n_bins = n_bins_list[ii]
    #         bins = np.linspace(np.min(np.concatenate([logE, logE_gen])), np.max(np.concatenate([logE, logE_gen])), n_bins + 1)
    #         bin_centers = 0.5 * (bins[:-1] + bins[1:])  
    #         cogy, loge, cogy_gen, loge_gen = [], [], [], []
            
    #         for i in range(n_bins): 
    #             in_bin = (logE >= bins[i]) & (logE < bins[i+1])
    #             in_bin_gen = (logE_gen >= bins[i]) & (logE_gen < bins[i+1])
    #             if in_bin.sum() == 0: continue
    #             else:
    #                 cogy.append(np.mean(new_cog[in_bin])) 
    #                 loge.append(np.mean(logE[in_bin]))  
    #                 cogy_gen.append(np.mean(new_cog_gen[in_bin_gen]))
    #                 loge_gen.append(np.mean(logE_gen[in_bin_gen]))
            
    #         plt.figure(11, figsize=(16, 25))
    #         j+=1
    #         plt.subplot(len(n_bins_list),2,j)
    #         plt.title("COG along y vs Log("+title[k]+"): bins --> "+str(n_bins), fontsize=font-10)
    #         plt.plot(loge, cogy, 'o-', color='k', label='Geant4')
    #         plt.plot(loge_gen, cogy_gen, 'o-', color='orange', label='CaloHadronic')
    #         if k==0: plt.legend(fontsize=font-10, loc='upper left')
    #         else: plt.legend(fontsize=font-10, loc='upper right')
    #         plt.xlabel("Log("+title[k]+")", fontsize=font-10)
    #         plt.ylabel("COG along y [layers]", fontsize=font-10)
    # plt.tight_layout()
    # plt.savefig(f"{my_dir}/Energies_vs_COGy.png")
    # plt.close()
    # sys.exit()

    thebins = [36, 36]
    H, xedges, yedges = np.histogram2d(to_plot , cog_y , bins=thebins, range=[_range0, _range1])
    H_r, _, _ = np.histogram2d(to_plot_r, cog_y_r, bins=thebins, range=[_range0, _range1])
    vmax = max(H.max(), H_r.max())
    
    fig = plt.figure(10, figsize=(32, 8.1))
    spec = gridspec.GridSpec(1, 3, width_ratios=[3.21, 4, 4], wspace=0.3)
    # First subplot
    ax1 = fig.add_subplot(spec[0, 0])
    hist1 = ax1.hist2d(
        to_plot, cog_y, bins=thebins, range=[_range0, _range1],
        vmin=vmin, vmax=vmax, cmap=color_cmap #, norm=mpl.colors.LogNorm()
    )
    ax1.set_title("CaloHadronic", fontsize=font, pad=20)
    ax1.set_xlabel("Energy Sum [MeV]", fontsize=font)
    ax1.set_ylabel("COG along y", fontsize=font)
    
    # Second subplot    
    ax2 = fig.add_subplot(spec[0, 1])
    hist2 = ax2.hist2d(
        to_plot_r, cog_y_r, bins=thebins, range=[_range0, _range1],
        vmin=vmin, vmax=vmax, cmap=color_cmap #, norm=mpl.colors.LogNorm()
    )
    ax2.set_title("Geant4", fontsize=font, pad=20)
    ax2.set_xlabel("Energy Sum [MeV]", fontsize=font)
    ax2.set_ylabel("COG along y", fontsize=font)
    
    # ax3 = fig.add_subplot(spec[0, 2])
    # ax3.axis('off')
    cbar1 = fig.colorbar(hist2[3], ax=ax2, orientation='vertical') #, fraction=0.42, location='left')
    cbar1.set_label("Counts", fontsize=font)
    cbar1.ax.yaxis.set_ticks_position('right')
    # cbar1.ax.yaxis.set_label_position('left')
       
    ax4 = fig.add_subplot(spec[0, 2])
    weighted_diff = np.transpose(H_r-H) 
    val = np.abs(weighted_diff).max()
    p4 = ax4.pcolormesh(xedges, yedges, weighted_diff, vmin=-val, vmax=val, cmap='PRGn')
    ax4.set_title(r"Geant4 - CaloHadronic", fontsize=font, pad=20)
    ax4.set_xlabel("Energy Sum [MeV]", fontsize=font)
    ax4.set_ylabel("COG along y", fontsize=font)
    ax4.set_ylim(_range1)
    ax4.set_xlim(_range0)
    
    # ax5 = fig.add_subplot(spec[0, 4])
    # ax5.axis('off')
    cb = fig.colorbar(p4, ax=ax4, orientation='vertical') #, fraction=0.42)
    cb.ax.yaxis.set_ticks_position('right')
    
    # tight_layout(fig) 
    plt.savefig(f"{my_dir}/Correlation_IncEn_COGy_histograms_"+str(thebins[0])+".pdf") 
    plt.close()
    
    
def plotting_correlations_withN(my_dir, fake, real, 
                                labels=['$E_{sum}$', 'N_{Hits}'], 
                                names=['Geant4', 'CaloHadronic']): # axis= 0,1,3
    _range0= [0, 2400]
    _range1= [0, 2500]  
    color_cmap = 'Oranges'
    vmin = 0
    to_plot = fake[0]     #sum over z and x 
    to_plot_r = real[0] #sum over z and x
    n = fake[1]
    n_r = real[1] 
    thebins = [36, 36]
    H, xedges, yedges = np.histogram2d(to_plot , n, bins=thebins, range=[_range0, _range1])
    H_r, _, _ = np.histogram2d(to_plot_r, n_r, bins=thebins, range=[_range0, _range1])
    vmax = max(H.max(), H_r.max())
    
    fig = plt.figure(10, figsize=(32, 8.1))
    spec = gridspec.GridSpec(1, 3, width_ratios=[3.21, 4, 4], wspace=0.3)
    # First subplot
    ax1 = fig.add_subplot(spec[0, 0])
    hist1 = ax1.hist2d(
        to_plot, n, bins=thebins, range=[_range0, _range1],
        vmin=vmin, vmax=vmax, cmap =color_cmap 
    )
    ax1.set_title("CaloHadronic", fontsize=font, pad=20)
    ax1.set_xlabel("Energy Sum [MeV]", fontsize=font)
    ax1.set_ylabel("Number of hits", fontsize=font)
    
    # Second subplot    
    ax2 = fig.add_subplot(spec[0, 1])
    hist2 = ax2.hist2d(
        to_plot_r, n_r, bins=thebins, range=[_range0, _range1],
        vmin=vmin, vmax=vmax, cmap =color_cmap #, norm=mpl.colors.LogNorm()
    )
    ax2.set_title("Geant4", fontsize=font, pad=20)
    ax2.set_xlabel("Energy Sum [MeV]", fontsize=font)
    ax2.set_ylabel("Number of Hits", fontsize=font)
    
    # ax3 = fig.add_subplot(spec[0, 2])
    # ax3.axis('off')
    cbar1 = fig.colorbar(hist2[3], ax=ax2, orientation='vertical') #, pad=-0.1, location='left')
    cbar1.set_label("Counts", fontsize=font)
    cbar1.ax.yaxis.set_ticks_position('right')
    # cbar1.ax.yaxis.set_label_position('left')
    
    ax4 = fig.add_subplot(spec[0, 2])
    weighted_diff = np.transpose(H_r-H) 
    val = np.abs(weighted_diff).max() 
    p4 = ax4.pcolormesh(xedges, yedges, weighted_diff, vmin=-val, vmax=val, cmap='PRGn')
    ax4.set_title(r"Geant4 - CaloHadronic", fontsize=font, pad=20)
    ax4.set_xlabel("Energy Sum [MeV]", fontsize=font)
    ax4.set_ylabel("Number of hits", fontsize=font)
    ax4.set_ylim(_range1)
    ax4.set_xlim(_range0)
    fig.colorbar(p4, ax=ax4, orientation='vertical')
    # fig.colorbar(hist4[3], ax=ax4, location='bottom')
     
    plt.savefig(f"{my_dir}/Correlation_N_histograms_"+str(thebins[0])+".pdf") 
    plt.close()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None): 
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return zs.mean()
        
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = clr.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plt_3dShowers(images, model_titles, save_titles, my_dir):
    vmin = 10.0
    vmax = 0.1
    
    for ima in images:
        vmin = min(vmin, np.min(ima[np.nonzero(ima)]))
        vmax = max(vmax, np.max(ima[np.nonzero(ima)]))
        
    print('vmin, vmax', vmin, vmax)
        
    for i in range(len(images)):
        plt_3dShower(images[i], model_title=model_titles[i], save_title=save_titles[i], my_dir=my_dir, vmax=vmax, vmin=vmin)
        
def plt_3dShower(image, model_title='ML Model', save_title='ML_model', my_dir='', vmax=None, vmin=None):
    
    if save_title+".png" == 'Shower_Real_50_GeV.png':
        figExIm = plt.figure(figsize=(20,16))
    else:
        figExIm = plt.figure(figsize=(16,16))

    axExIm1 = figExIm.add_subplot(projection='3d')
    image = image+0.0
    
    masked_array = np.ma.array(image, mask=(image<=0.005))
    cmap = mpl.cm.viridis
    axExIm1.view_init(elev=20.0, azim=20.0)
    xL,yL,zL,cL = [],[],[],[]
    for index, c in np.ndenumerate(masked_array):
        (x,y,z) = index
        if c != 0:
            xtmp = x
            if xtmp%2 == 0:
                xtmp = xtmp + 0
            else:
                xtmp = xtmp - 0

            xL.append(xtmp)
            yL.append(y)
            zL.append(z)
            cL.append(c)

    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)

    xL = np.array(xL)
    yL = np.array(yL)
    zL = np.array(zL)
    cL = np.array(cL)
    figExIm.patch.set_facecolor('white')
    
    cmap = mpl.cm.jet
    my_cmap = truncate_colormap(cmap, 0.0, 0.7)
    transparent = (0.1, 0.1, 0.9, 0.0)
    
    axExIm1.set_xticklabels([])
    axExIm1.set_yticklabels([])
    axExIm1.set_zticklabels([])
    limit_shift = 30
    axExIm1.set_xticks(np.linspace(xL.mean() -limit_shift , xL.mean() +limit_shift , 78))
    axExIm1.set_yticks(np.arange(0, 78, 1))
    axExIm1.set_zticks(np.linspace(zL.mean() -limit_shift , zL.mean() +limit_shift , 78))    
    
    axExIm1.set_xlabel('x [cells]', family='serif', fontsize='35')
    axExIm1.set_ylabel('y [layers]', family='serif', fontsize='35')
    axExIm1.set_zlabel('z [cells]', family='serif', fontsize='35')

    axExIm1.set_xlim([xL.mean() -limit_shift , xL.mean() +limit_shift])
    axExIm1.set_ylim([0, 78])
    axExIm1.set_zlim([zL.mean() -limit_shift , zL.mean() +limit_shift])
    
    xx, zz = np.meshgrid([xL.mean() -limit_shift, xL.mean() +limit_shift], [zL.mean() -limit_shift, zL.mean() +limit_shift])
    distance_from_center = np.sqrt(xx**2 + zz**2)
    norm = (distance_from_center - distance_from_center.min()) / (distance_from_center.max() - distance_from_center.min())
    axExIm1.plot_surface(xx, [[30,30], [30,30]], zz, alpha=0.1, facecolors=plt.cm.Reds_r(norm), shade=False, rstride=1, cstride=1, antialiased=False)
    axExIm1.grid(True)
    axExIm1.xaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.2, alpha=0.2)
    
    to_zero_values = ((xL== xL.mean()-limit_shift) | (xL== xL.mean()+limit_shift)) | ((zL== zL.mean()-limit_shift) | (zL== zL.mean()+limit_shift))
    cL[to_zero_values] = 0
    
    a = Arrow3D([xL.mean(), xL.mean() ], [-15, -2], 
                [zL.mean(), zL.mean()], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="k")
    axExIm1.add_artist(a)
    
    plotMatrix(ax=axExIm1, x=xL, y=yL, z=zL, data=cL, cmap=my_cmap, alpha=0.7, edgecolors=transparent, vmax=vmax, vmin=vmin)    
    if save_title+".png" == 'Shower_Real_50_GeV.png':
        norm = mpl.colors.LogNorm(vmin=0.099, vmax=vmax)
        cbar = figExIm.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), ax=axExIm1)
        cbar.ax.tick_params(labelsize='35') 

    plt.savefig(my_dir + '/'+ save_title+".png")
    
def plotMatrix(ax, x, y, z, data, cmap="jet", cax=None, alpha=0.1, edgecolors=None, vmax=None, vmin=None):
    # plot a Matrix 
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    colors = lambda i : mpl.cm.ScalarMappable(norm=norm, cmap = cmap).to_rgba(data[i]) 
    norm_max = vmax

    for i, xi in enumerate(x):
        # alp2 = 0.1+0.9*np.log(data[i]*10)/np.log(norm_max*10)
        alp2 = 0.3
        plotCubeAt(pos=(x[i], y[i], z[i]), l=(0.1, 0.8, 0.8), c=colors(i), c2=colors(i), alpha=alp2,  ax=ax, edgecolors=edgecolors)
   
def plotCubeAt(pos=(0,0,0), l=(1.0,1.0,1.0), c="b", c2="k", alpha=0.1, ax=None, edgecolors=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        x_range = np.array([[pos[0], pos[0]+l[0]]])
        y_range = np.array([[pos[1], pos[1]+l[1]]])
        z_range = np.array([[pos[2], pos[2]+l[2]]])
        
        z_range
        xx, yy = np.meshgrid(x_range, y_range)
        
        ax.plot_surface(xx, yy, (np.tile(z_range[:,0:1], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, yy, (np.tile(z_range[:,1:2], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        lw=0.5
        
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,0:1], (2)), zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,1:2], (2)), zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,0:1], (2)), zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,1:2], (2)), zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        
        yy, zz = np.meshgrid(y_range, z_range)
        ax.plot_surface((np.tile(x_range[:,0:1], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface((np.tile(x_range[:,1:2], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=y_range[0], zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=y_range[0], zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=y_range[0], zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=y_range[0], zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        
        xx, zz = np.meshgrid(x_range, z_range)
        ax.plot_surface(xx, (np.tile(y_range[:,0:1], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, (np.tile(y_range[:,1:2], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
    
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=np.tile(y_range[0,0:1], (2)), zs=z_range[0], lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=np.tile(y_range[0,0:1], (2)), zs=z_range[0], lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=np.tile(y_range[0,1:2], (2)), zs=z_range[0], lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=np.tile(y_range[0,1:2], (2)), zs=z_range[0], lw=lw, c=c2)


