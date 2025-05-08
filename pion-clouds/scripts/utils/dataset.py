from torch.utils.data import Dataset
import numpy as np
# import h5py
import h5pickle as h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm
from pathlib import Path
import os

class PionClouds(Dataset):
    def __init__(self, files_path, bs=32, quantized_pos=True, only_hcal=False, only_ecal=False):
        self.Ymin, self.Ymax = 0, 78
        if only_hcal: self.Ymin, self.Ymax = 30, 78
        elif only_ecal: self.Ymin, self.Ymax = 0, 30
         
        self.Xmin, self.Xmax = -450, 450
        self.Zmin, self.Zmax = -450, 450
            
        self.bs = bs  
        self.only_hcal = only_hcal
        self.only_ecal = only_ecal
        
        self.max_n_points = 5000 
        if (self.only_hcal) | (self.only_ecal): self.max_n_points = 3200
        
        # create a sorted index and open data files 
        index_list = []
        self.data_files = []
        self.n_files_in_the_folder = len(os.listdir(Path(files_path).parent))
         
        for file_idx in range(1, self.n_files_in_the_folder+1):
            # if file_idx%20==0: print('load file ', file_idx, '    |', end=" ")
            
            f = h5py.File(files_path.format(file_idx), 'r')
            self.data_files.append(f)
              
            n_points = f['n_points'][:]
             
            for i, n_point in enumerate(n_points):
                if n_point < 20:
                    continue
                index_list.append((n_point, file_idx-1, i)) 
        
        # sort the index list by 'n_points'
        self.index_list = sorted(index_list, key=lambda x: x[0])
            
    def _getbatch(self, idx):
        idx_range = self._determine_idx_range(idx) #(min_range, max_range)
        max_num_points = self.index_list[idx_range[1]-1][0] # max number of points in the batch
        events = []
        energy = []
        n_points = []
        for i in range(idx_range[0], idx_range[1]):
            n_point, file_idx, event_idx = self.index_list[i]
            f = self.data_files[file_idx]

            event_data = f['events'][event_idx] 
            n_points.append(f['n_points'][event_idx]) 
            events.append(event_data[:, -int(max_num_points):]) 
            energy.append(f['energy'][event_idx])       

        events = np.array(events)
        energy = np.array(energy)
        n_points = np.array(n_points)
        return events, energy, n_points

    def _determine_idx_range(self, idx):
        if idx > self.bs and idx < len(self) - self.bs:
            return (idx - int(self.bs / 2), idx + int(self.bs / 2) + 1)
        elif idx < self.bs:
            return (idx, idx + self.bs)
        else:
            return (idx - self.bs, idx)   
    
    def __getitem__(self, idx):
        events, energy, n_points = self._getbatch(idx)
        
        if (self.only_hcal==False) & (self.only_ecal==False):
            padding_mask = np.zeros((events.shape[0], events.shape[-1])).astype(bool)
            col_indices = np.flip(np.arange(events.shape[-1]))
            padding_mask[col_indices >= np.array(n_points)[:, None]] = True 
        else: padding_mask = np.zeros((events.shape[0], events.shape[-1])) 
         
        events[:, 3, :] *= 1000  # energy scale
        events[:, 0, :] = (events[:, 0, :] - self.Xmin) * 2 / (self.Xmax - self.Xmin) - 1  # x normalization
        events[:, 2, :] = (events[:, 2, :] - self.Zmin) * 2 / (self.Zmax - self.Zmin) - 1  # z normalization
          
        events = np.moveaxis(events, -1, -2)
        events[events[:, :, 3] == 0] = 0
           
        if self.only_hcal or self.only_ecal:
            events, cond_ecal, padding_mask_hcal, padding_mask_ecal = self.get_only_hcal_data(events) 
            if self.only_hcal:
                padding_mask = padding_mask_hcal     
                n_points = self._calculate_n_points(events)
                # smearing propagation axis
                smearing = np.random.uniform(-0.49, 0.49, size=cond_ecal[:, :, 1].shape)
                cond_ecal[:, :, 1] = cond_ecal[:, :, 1] + smearing
                cond_ecal[:, :, 1] = (cond_ecal[:, :, 1] - 0) * 2 / (30 - 0) - 1 
                cond_ecal = cond_ecal[:,:,[0,1,2,3]]
            else:
                padding_mask = padding_mask_ecal 
                events = cond_ecal 
                n_points = self._calculate_n_points(cond_ecal)

            if (~padding_mask).sum(axis=1).min()==0:
                w = np.argwhere((~padding_mask).sum(axis=1)==0) 
                print('WARNING: no hits in one or more showers in the batch --> reducing batch size of ', len(w))
                events = np.delete(events, w, axis=0)
                padding_mask = np.delete(padding_mask, w, axis=0)
                cond_ecal = np.delete(cond_ecal, w, axis=0)
                energy = np.delete(energy, w, axis=0)

            # ordering the points based on the energy, so that padded points are at the end 
            # idx_sorted = np.flip(np.argsort(events[:,:,3], axis=1), axis=1) #sorting in descending order
            # for i in range(4): events[:, :, i] = events[:, :, i][np.arange(events.shape[0])[:, None], idx_sorted]
            # padding_mask = padding_mask[np.arange(padding_mask.shape[0])[:, None], idx_sorted] 
        else: 
            cond_ecal = np.zeros(events.shape)
        
        points_per_layer = self._calculate_points_per_layer(events, axis=1, only_hcal=self.only_hcal, only_ecal = self.only_ecal)
        max_PperL = points_per_layer.max()
        points_per_layer = points_per_layer / max_PperL # now I normalize the entire batch 
        
        # smear on the propagation axis
        smearing = np.random.uniform(-0.49, 0.49, size=events[:, :, 1].shape)
        events[:, :, 1] = events[:, :, 1] + smearing 
             
        events[:, :, 1] = (events[:, :, 1] - self.Ymin) * 2 / (self.Ymax - self.Ymin) - 1  # y normalization
        events = events[:, :, [0,1,2,3]] # x, y, z, energy
        n_points = n_points[:, None] / self.max_n_points 
        
        # bb = 50
        # c = cond_ecal[:,:,3] > 0 
        # plt.figure(1)
        # plt.subplot(3,2,1)
        # plt.hist(energy.flatten(), bins=bb)
        # plt.subplot(3,2,2)
        # plt.hist(points_per_layer.flatten(), bins=bb)
        # plt.subplot(3,2,3)
        # plt.hist(cond_ecal[:,:,0][c].flatten(), bins=bb)
        # plt.subplot(3,2,4)
        # plt.hist(cond_ecal[:,:,1][c].flatten(), bins=bb)
        # plt.subplot(3,2,5)
        # plt.hist(cond_ecal[:,:,2][c].flatten(), bins=bb)
        # plt.subplot(3,2,6)
        # plt.hist(np.log(cond_ecal[:,:,3][c].flatten()+1e-20), bins=bb)
        # plt.savefig('cond_feats_train.png')
        # sys.exit()
        
        # if self.only_hcal:
        #     idx = np.argsort(cond_ecal[:,:,1], axis=1)
        #     cond_ecal = np.take_along_axis(cond_ecal, idx[:, :, np.newaxis], axis=1)  # Efficient batch sorting
        #     tot_points = cond_ecal.shape[1]
        #     limit_points = 200 
        #     if tot_points > limit_points: 
        #         _list = np.linspace(0, tot_points, limit_points).astype(int)
        #         new_cond_ecal = np.zeros((cond_ecal.shape[0], limit_points, cond_ecal.shape[2])) 
        #         for k in range(_list.shape[0]-1): 
        #             # mean of the coordinates 
        #             new_cond_ecal[:,k,:3] = np.mean(cond_ecal[:, _list[k]:_list[k+1], :3], axis=1)
        #             # sum of energy
        #             new_cond_ecal[:,k,3] = np.sum(cond_ecal[:, _list[k]:_list[k+1], 3], axis=1)
        #         cond_ecal = new_cond_ecal     
        #         del new_cond_ecal
                
        return {
            'event': events,
            'energy': energy,
            'points_per_layer': points_per_layer,
            'points_per_layer_max': max_PperL,
            'n_points': n_points,
            'cond_ecal': cond_ecal,
            'padding_mask': padding_mask,
        }

    def _calculate_n_points(self, events):
        return np.sum(events[:,:,3]>0, axis=1)

    def _calculate_points_per_layer(self, events, axis=1, only_hcal=False, only_ecal=False):
        points_per_layer = []
        if axis==1: min_range, max_range, bins = 0, 78, 78
        else: min_range, max_range, bins = -1, 1, 30
        
        for event in events:
            num_p, _ = np.histogram(event[:, axis][event[:, 3] > 0], bins=bins, range=(min_range, max_range))
            if (only_hcal) & (axis==1): num_p = num_p[30:]
            if (only_ecal) & (axis==1): num_p = num_p[:30]
            points_per_layer.append(num_p) #/ num_p.max())
        return np.array(points_per_layer)
   
    def get_only_hcal_data(self,events):
        # use only hcal and ecal part as conditioning
        dim_a, dim_b, dim_c = events.shape
        events2 = events.copy()
        condition_ecal = (events2[:, :, 1] < 30) & (events2[:, :, 3] > 0) 
        max_point_ecal = np.sum(condition_ecal, axis=1).max()
        max_point_hcal = np.sum(events2[:, :, 1] >= 30, axis=1).max()
         
        #re-do padding mask. Here is needed because the points are not ordered by n_points in the shower anymore!
        padding_mask_hcal = np.zeros((dim_a, max_point_hcal)).astype(bool)
        padding_mask_ecal = np.zeros((dim_a, max_point_ecal)).astype(bool) 
        col_indices_hcal = np.arange(max_point_hcal)
        col_indices_ecal = np.arange(max_point_ecal)
        
        events, cond_ecal = np.zeros((dim_a, max_point_hcal, dim_c)) , np.zeros((dim_a, max_point_ecal, dim_c))
        for j in range(events2.shape[0]):
            condition_ecal = (events2[j, :, 1] < 30) & (events2[j, :, 3] > 0)
            to_concatenate = max_point_ecal - np.sum(condition_ecal) 
            cond_ecal[j] = np.concatenate((events2[j, condition_ecal], np.zeros((to_concatenate, dim_c))), axis=0) 
            padding_mask_ecal[j, col_indices_ecal >= np.sum(condition_ecal) ] = True       
            
            to_concatenate = max_point_hcal - np.sum(events2[j, :, 1] >= 30) 
            events[j] = np.concatenate((events2[j, events2[j, :, 1] >= 30], np.zeros((to_concatenate, dim_c))), axis=0)
            padding_mask_hcal[j, col_indices_hcal >= np.sum(events2[j, :, 1] >= 30)] = True       
        del events2 
        return events, cond_ecal, padding_mask_hcal, padding_mask_ecal
   
    def __len__(self):
        return len(self.index_list)

