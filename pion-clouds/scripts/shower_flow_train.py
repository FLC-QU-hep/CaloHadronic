# import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
import torch.utils.data
from torch import optim
import h5py
import time
from utils.misc import *
import matplotlib
import matplotlib.pyplot as plt
import sys
from models.shower_flow import compile_HybridTanH_model, FlowMatchingLoss
import models.flow_matching as fm
import logging
import os
import shutil
from scipy.stats import wasserstein_distance
from adam_mini import Adam_mini

# Load the config.yaml file
cfg = OmegaConf.load('configs/configs_sf.yaml')
seed_all(seed = cfg.seed)
print(cfg)

outdir = cfg.ckps_dir + '/shw_log_dir_'+cfg.name+'/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Copy the config.yaml file to the output directory
shutil.copy('configs/configs_sf.yaml', os.path.join(outdir, 'configs_sf.yaml'))

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(outdir + '/shower_flow_train_large_'+cfg.name+'.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

device = torch.device(cfg.device)

n = cfg.name.split('_')[0]
points_per_layer = np.load(cfg.load_dir + 'points_'+n+'/n_points_per_layer_for_SF.npy')
cond_energy = np.load(cfg.load_dir +'energy_for_SF_'+n+'.npy')
# n_points_ecal = np.load(cfg.load_dir +'points_HGx4_OnlyECAL/n_points_for_SF.npy')
energy_per_layer = np.load(cfg.load_dir +'points_'+n+'/energy_per_layer_for_SF.npy')

encoding = np.zeros(points_per_layer.shape)
if cfg.hot_encoding!=0:
    # even-odd 
    idx_even = np.arange(0, 78 , 2)
    idx_odd = np.arange(1, 79 , 2)
    encoding[:, idx_even] = 1
    print('even-odd done with HOT ENCODING')    

if cfg.model.CNF:
    model = fm.CNF(fm.FullyConnected(**cfg.fm))
else:
    model, distribution = compile_HybridTanH_model(
        num_blocks=cfg.model.num_blocks, 
        num_inputs=cfg.fm.num_inputs,
        num_cond_inputs=cfg.fm.num_cond_inputs, 
        device=cfg.device,
        input_dim_multiplier=cfg.model.input_dim_multiplier
    )  

logger.info('Model compiled')
# num_params = sum(p.numel() for p in model.parameters())
# logger.info(f'Number of parameters: {num_params}')

print('MODEL INFO:')
print('lr: ', cfg.model.lr,'    |    num_blocks: ', cfg.model.num_blocks, '    |    num_inputs: ', cfg.fm.num_inputs)
print('num_cond_inputs: ', cfg.fm.num_cond_inputs, '    |    input_dim_multiplier: ', cfg.model.input_dim_multiplier)

logger.info('Data loaded')
print(cond_energy.reshape(-1).shape, points_per_layer.shape)

# pre-proc points per layer: log so in generation it geneates only pos values
pp = np.log(points_per_layer+1e-20)
p_per_l_max = np.max(pp)
print('p_per_l_max in log: ', p_per_l_max)

dataset = torch.utils.data.TensorDataset(
    torch.tensor(encoding),
    torch.tensor(cond_energy.reshape(-1)/cond_energy.max()), # scaled conditional energy 
    torch.tensor(pp / p_per_l_max), # scaled num points per layer
    )
print(p_per_l_max)
print(cfg.name)

print('n max:', points_per_layer.sum(axis=1).max(),' |  e max: ', cond_energy.max())
print('|  p per l max: ', points_per_layer.max(), '|  e per l max: ', energy_per_layer.max())

train_size = int(cfg.model.perc_train *points_per_layer.shape[0])
train_batch, val_batch = int(cfg.model.perc_train*cfg.model.batch_size), int(cfg.model.perc_val*cfg.model.batch_size)
print('batch sizes: ', train_batch, val_batch)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, points_per_layer.shape[0]-train_size])

val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=val_batch, 
        shuffle=cfg.model.shuffle, 
        pin_memory=cfg.model.pin_memory,
        num_workers=cfg.model.num_workers
    )   

train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch, 
        shuffle=cfg.model.shuffle,
        pin_memory=cfg.model.pin_memory,
        num_workers=cfg.model.num_workers
    )
       
outpath = outdir
model.to(device) 
if cfg.optimizer=='Adam':
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=cfg.model.lr, 
                total_steps=cfg.model.epochs*len(train_loader)
            )
elif cfg.optimizer=='Adam-mini':
        optimizer = Adam_mini(
            named_parameters = model.named_parameters(),
            lr = cfg.model.lr,
            betas = (0.9, 0.999), #same as Adam
            eps = 1e-08, #same as Adam
            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=cfg.model.lr, 
                total_steps=cfg.model.epochs*len(train_loader),
                cycle_momentum=False
            )
else:
    raise ValueError(f'Optimizer {cfg.optimizer} not defined')

logger.info('Training started')
print('Training started')
losses, val_losses, wdist_list, lr = [], [], [], []
epoch_start = 1
wd_list = []
for epoch in range(epoch_start, cfg.model.epochs +1):
    input_list = []
    train_loss_list, val_loss_list = [], []
    
    # Training
    model.train()
    for batch_idx, (encoding, energy, clusters_per_layer) in enumerate(train_loader):
        encoding = encoding.to(device).float()
        E_true = energy.view(-1, 1).to(device).float()
        clusters_per_layer1 = clusters_per_layer.to(device).float()
        input_data = clusters_per_layer1
        
        optimizer.zero_grad()
        if cfg.hot_encoding!=0: 
            context = torch.cat((E_true, encoding), 1)
        else: context = E_true 

        if np.any(np.isnan(input_data.clone().detach().cpu().numpy())) == True:
            logger.info('Nans in the training data!')
            print('Nans in the training data!')

        if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any():
            logger.info('Weights are nan!')
            print('Weights are nan!')
            if epoch>1:
                # load recent model
                model.load_state_dict(torch.load(outpath+f'ShowerFlow_latest.pth')['model'])
                optimizer = optim.Adam(model.parameters(), lr=cfg.model.lr)
                print(f'latest model reloaded, optimizer resetted')
                logger.info(f'latest model reloaded, optimizer resetted')
            else: 
                for p in model.parameters(): 
                    if torch.isnan(p).any(): p.data = torch.ones_like(p.data)
                print('weights are nan in the first epoch, setting them to onw')
        
        if cfg.model.CNF: 
            nll = model.loss(input_data, condition=context)
        else: 
            nll = -distribution.condition(context).log_prob(input_data) 
        loss = nll.mean() 
        loss.backward() 
        optimizer.step()  
        scheduler.step()
        train_loss_list.append(loss.item())      
    
    losses.append(np.mean(train_loss_list))
    lr.append(scheduler.get_last_lr()[0])
    # Validation 
    model.eval()
    for batch_idx, (encoding, energy, clusters_per_layer) in enumerate(val_loader): 
        encoding = encoding.to(device).float()
        E_true = energy.view(-1, 1).to(device).float() 
        clusters_per_layer2 = clusters_per_layer.to(device).float() 
        input_data = clusters_per_layer2
        
        if cfg.hot_encoding!=0: 
            context_val = torch.cat((E_true, encoding), 1)
        else: context_val = E_true 
        
        if cfg.model.CNF: 
            nll = model.loss(input_data, condition=context_val)
        else: 
            nll = -distribution.condition(context_val).log_prob(input_data)       
        loss = nll.mean() 
        loss.backward() 
        val_loss_list.append(loss.item())                    
    
    val_losses.append(np.mean(val_loss_list))
    print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss_list):.2f}, Val Loss: {np.mean(val_loss_list):.2f}')
    logger.info(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss_list):.2f}, Val Loss: {np.mean(val_loss_list):.2f}')
    
    # plots 
    plt.figure(1)
    plt.plot(losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()    
    plt.savefig(outpath+f'losses.png')
    if epoch == 600: plt.savefig(outpath+f'losses_'+str(epoch)+'.png')
    plt.close()
    plt.figure(1)
    plt.plot(lr, label='lr')
    plt.legend()    
    plt.savefig(outpath+f'lr.png')
    plt.close()   
     
    if epoch == 1: best_loss = np.mean(val_loss_list) #best is when the validaion is best

    if torch.stack([torch.isnan(p).any() for p in model.parameters()]).any(): # save models only if no nan weights
        print('model not saved due to nan weights')
        logger.info('model not saved due to nan weights')
    else:
        save_model = True
        torch.save(
            {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),},
            outpath+f'ShowerFlow_latest.pth'
        )
        if epoch%10 == 0:
            torch.save(
                {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),},
                outpath+f'ShowerFlow_{epoch}.pth'
            )
        # save best model based on loss
        if np.mean(val_loss_list) <= best_loss:
            best_loss = np.mean(val_loss_list)
            torch.save(
                {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),},
                outpath+f'ShowerFlow_bestLoss.pth'
            )

            context = torch.concatenate((context, context_val), dim=0)
            points_per_layer = torch.concatenate((clusters_per_layer1, clusters_per_layer2), dim=0).detach().cpu().numpy()
            checkpoint = torch.load(outpath+'/ShowerFlow_bestLoss.pth', map_location=torch.device(device))
            model.load_state_dict(checkpoint['model']) 
            
            if cfg.model.CNF:
                with torch.no_grad():
                    samples = model.sample(
                        (context.shape[0], cfg.fm.num_inputs), condition=context
                    ).cpu().numpy()
            else:
                samples = distribution.condition(context).sample(torch.Size([context.shape[0], ])).cpu().numpy()
            
            # post-process samples
            clusters_per_layer = np.exp(samples[:, :78] * p_per_l_max).astype(int) 
            points_per_layer = np.exp(points_per_layer * p_per_l_max).astype(int)   
            _rangeECAL = (np.nanmin(points_per_layer[:, :30].flatten()), np.nanmax(points_per_layer[:, :30].flatten()))
            _rangeHCAL = (np.nanmin(points_per_layer[:, 30:].flatten()), np.nanmax(points_per_layer[:, 30:].flatten())) 
            wd_arr = np.zeros((context.shape[0],))
            plt.figure(2, figsize=(35, 15))
            for layer in range(78):
                wd_arr[layer] = wasserstein_distance(points_per_layer[:, layer].flatten(), 
                                                clusters_per_layer[:, layer].flatten())
                _range = _rangeECAL if layer < 30 else _rangeHCAL
                plt.subplot(6, 13, layer+1) 
                plt.hist(points_per_layer[:, layer], bins=np.logspace(np.log10(1e-1), np.log10(1e3), 40), range=_range, alpha=0.9, color='gray', label='geant4')
                plt.hist(clusters_per_layer[:, layer], bins=np.logspace(np.log10(1e-1), np.log10(1e3), 40), range=_range, histtype='step', color='g', label='CaloHadronic')
                plt.xlabel('Points in Layer '+str(layer), fontsize=12)
                if layer%10==0: plt.legend()
                plt.yscale('log')
                plt.xscale('log')
            plt.tight_layout()
            plt.savefig(outpath+f'distribution.png')
            if epoch%50 == 0: plt.savefig(outpath+f'distribution_'+str(epoch)+'.png') 
            plt.close() 
                            
            plt.figure(2, figsize=(8, 8))
            _range = (0, 2900)
            plt.hist(points_per_layer.sum(axis=1), bins=40, range=_range, alpha=0.9, color='gray', label='geant4')
            plt.hist(clusters_per_layer.sum(axis=1), bins=40, range=_range, histtype='step', color='g', linewidth=3, label='sum')
            plt.xlabel('Total Points ', fontsize=12)
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.savefig(outpath+f'hits_tot.png')
            plt.close()    
            print('plots done')
            
            
            