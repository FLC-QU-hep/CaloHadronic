import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchmetrics
from torchinfo import summary
from sklearn.isotonic import IsotonicRegression
import random
import h5py 
from tqdm import tqdm 
import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.clas_utils import *

class DenseClassifier(nn.Module):

    def __init__(self, in_features, hidden_features, use_leaky_relu):
        super().__init__()

        if use_leaky_relu:
            Activation = nn.LeakyReLU
        else:
            Activation = nn.ReLU

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            Activation(),
            nn.Linear(hidden_features, hidden_features),
            Activation(),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    

def get_datasets(obs1, obs2, cutoff:float=1e-4):
    obs = torch.cat((obs1, obs2)).to(torch.get_default_dtype())
    num_samples = len(obs1)
    label1 = torch.zeros(num_samples,1)
    label2 = torch.ones(num_samples,1)
    label = torch.cat((label1, label2))

    r = torch.randperm(len(obs))
    obs = obs[r]
    label = label[r]
    to_split = [0.6,0.2,0.2]
    to_train, to_val = int(len(obs)*to_split[0]), int(len(obs)*to_split[1])
    split = [to_train, to_val, len(obs)- to_train - to_val]
    obs_train, obs_val, obs_test = torch.split(obs, split)
    label_train, label_val, label_test = torch.split(label, split)
    
    mean = torch.mean(obs_train, dim=0, keepdim=True)
    std = torch.std(obs_train, dim=0, unbiased=False, keepdim=True)
    obs_train -= mean
    obs_train /= std
    obs_val -= mean
    obs_val /= std
    obs_test -= mean
    obs_test /= std

    train_data = TensorDataset(obs_train, label_train)
    val_data = TensorDataset(obs_val, label_val)
    test_data = TensorDataset(obs_test, label_test)
    return train_data, val_data, test_data 

def get_dataloaders(obs1, obs2, cut=1e-4, batch_size=256):
    train_dataset, val_dataset, test_data = get_datasets(obs1, obs2, cut)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=32
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2**11,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )
    test_loader = DataLoader(
        test_data,
        batch_size=2**11,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )
    return train_loader, val_loader, test_loader
  
def train_epoch(model, data, device, optimizer, criterion):
    model.train()
    average_loss = 0
    for obs, label in data:
        obs = obs.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        scores = model(obs)
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()

        average_loss += len(obs)*loss.item()
    average_loss /= len(data.dataset)
    return average_loss

@torch.no_grad()
def validation(model, data, device, criterion):
    model.eval()

    val_loss = 0
    all_label = []
    all_scores = []

    for obs, label in data:
        obs = obs.to(device)
        label = label.to(device)

        scores = model(obs)
        loss = criterion(scores, label)

        val_loss += len(obs)*loss.item()
        all_label.append(label)
        all_scores.append(torch.sigmoid(scores))

    all_label = torch.cat(all_label)
    all_scores = torch.cat(all_scores)
    val_loss /= len(data.dataset)

    return all_scores, all_label, val_loss

def train(model, obs1, obs2, lr=1e-3, batch_size=256, epochs=30, device='cpu', cutoff=1e-4):
    criterion = nn.BCEWithLogitsLoss() 
    criterion = criterion.to(device)  
    model = model.to(device)   
    optimizer = optim.Adam(model.parameters(), lr) 
    train_loader, val_loader, test_loader = get_dataloaders(obs1, obs2, cutoff, batch_size)
    print(train_loader, val_loader, test_loader)
    evaluate_metrics = EvaluateMetrics(device)

    to_numpy = lambda x: x.to(dtype=torch.float64, device='cpu').flatten().numpy()
    to_torch = lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=device).unflatten(0,(-1,1))
    tr_loss, vl_loss, t_loss = [], [], []
    scores_val, labels_val, scores_test_, labels_test_ = [], [], [], []
    for i in tqdm(range(epochs)):
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        scores, labels, val_loss = validation(model, val_loader, device, criterion)
        evaluate_metrics(scores, labels, train_loss=train_loss, val_loss=val_loss, name=f'epoch {i:3d}')
        calibrator = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6)
        calibrator.fit(to_numpy(scores), to_numpy(labels))
        scores_test, labels_test, test_loss = validation(model, test_loader, device, criterion)
        rescaled_scores = to_torch(calibrator.predict(to_numpy(scores_test)))
        evaluate_metrics(rescaled_scores, labels_test, test_loss=test_loss, name='test')
        tr_loss.append(train_loss) 
        vl_loss.append(val_loss)  
        t_loss.append(test_loss) 
        plotting_losses(tr_loss, vl_loss, t_loss, dir_name='high_level')
        scores_val.append(scores)
        labels_val.append(labels)
        scores_test_.append(rescaled_scores)
        labels_test_.append(labels_test)
        evaluate_metrics.plotting_metric(scores_val, labels_val, scores_test_, labels_test_, dir_name='high_level')
        
def main():
    prefix='HGx9_CNF_ECAL_HCAL_30'
    shw_list = [50000, 50001, 50002]
    inc_en = '' #default
    
    f_names = ['e_sum_list', 'occ_list', 'cog_x', 'cog_y', 'cog_z', 'inc_en'] 
    f_names_r=['e_sum_list_r', 'occ_list_r','cog_x_r', 'cog_y_r', 'cog_z_r', 'inc_en'] 
    tensor_real, tensor_fake = [], []
    
    dir_ = '../../files/projected_array/'
    
    for shw in shw_list:
        save_dir_fake = dir_+prefix+'/'+inc_en+str(shw) 
        save_dir_real = dir_+'Geant4/'+inc_en+str(shw)

        inc_energy_r = h5py.File(save_dir_real+'/events_'+str(shw)+'.hdf5', 'r')['inc_energy'][:] *1000
        inc_energy = h5py.File(save_dir_fake+'/events_'+str(shw)+'.hdf5', 'r')['inc_energy'][:] *1000

        real_dict = {}    
        for k, name in enumerate(f_names_r): 
            real_dict[name] = np.load(save_dir_real+'/'+name+'.npy')
        real_dict['en_ratio'] = np.array(real_dict['e_sum_list_r']) / real_dict['inc_en'].reshape(-1)
        del real_dict['inc_en']
        del real_dict['e_sum_list_r']
        array = np.moveaxis(np.array(list(real_dict.values())),-1,-2)
        tensor_real.append(torch.tensor(array))
    
        fake_dict = {}
        for k, name in enumerate(f_names): 
            fake_dict[name] = np.load(save_dir_fake+'/'+name+'.npy')
        fake_dict['en_ratio'] = fake_dict['e_sum_list'] / fake_dict['inc_en'].reshape(-1)
        del fake_dict['inc_en']
        del fake_dict['e_sum_list'] 
        array = np.moveaxis(np.array(list(fake_dict.values())),-1,-2)
        tensor_fake.append(torch.tensor(array))

    tensor_real = torch.cat(tensor_real, dim=0) #[torch.randperm(len(tensor_real))]
    tensor_fake = torch.cat(tensor_fake, dim=0) #[torch.randperm(len(tensor_fake))] 
    
    print('classifier type: high  level')
    model = DenseClassifier(
        in_features= tensor_fake.shape[1],
        hidden_features=4, 
        use_leaky_relu=True
    )
    
    print(model)
    print('')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model, tensor_real, tensor_fake,
        device=device,
        lr=0.001,
        batch_size=32,
        epochs=100)

if __name__=='__main__':
    main()
    
    