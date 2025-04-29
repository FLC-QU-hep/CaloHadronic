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

class EvaluateMetrics:
    def __init__(self, device):
        self.calc_auc = torchmetrics.AUROC(task='binary')
        self.calc_accuracy = torchmetrics.Accuracy(task='binary')
        self.calc_auc = self.calc_auc.to(device)
        self.calc_accuracy = self.calc_accuracy.to(device)
    
    @torch.no_grad()
    def plotting_metric(self, all_scores_val, all_label_val, all_scores_test, all_label_test, dir_name=''):
        auc_val, acc_val, jsd_val, jsd2_val = [], [], [], []
        auc_test, acc_test, jsd_test, jsd2_test = [], [], [], []
        
        for l in range(len(all_scores_val)):
            auc_val.append(self.calc_auc(all_scores_val[l], all_label_val[l]).item())
            acc_val.append(self.calc_accuracy(all_scores_val[l], all_label_val[l]).item())
            jsd_val.append(calc_JSD(all_scores_val[l], all_label_val[l]).item())
            jsd2_val.append(calc_JSD(all_scores_val[l]).item())
            auc_test.append(self.calc_auc(all_scores_test[l], all_label_test[l]).item())
            acc_test.append(self.calc_accuracy(all_scores_test[l], all_label_test[l]).item())
            jsd_test.append(calc_JSD(all_scores_test[l], all_label_test[l]).item())
            jsd2_test.append(calc_JSD(all_scores_test[l]).item())
        
        plt.figure(1, figsize=(10,10))
        plt.subplot(2,2,1)
        plt.plot(auc_val, label='auc val')
        plt.plot(auc_test, label='auc test')
        plt.legend()
        plt.subplot(2,2,2)
        plt.plot(acc_val, label='acc val')
        plt.plot(acc_test, label='acc test')
        plt.legend() 
        plt.subplot(2,2,3)
        plt.plot(jsd_val, label='jsd val')
        plt.plot(jsd_test, label='jsd test')
        plt.legend() 
        plt.subplot(2,2,4) 
        plt.plot(jsd2_val, label='jsd2 val')
        plt.plot(jsd2_test, label='jsd2 test')
        plt.legend()  
        plt.savefig("/data/dust/user/mmozzani/pion-clouds/figs/classifier/"+dir_name+"/auc_jsd.png")
        plt.close()
    
    @torch.no_grad()
    def __call__(self, all_scores, all_label, train_loss=None, val_loss=None, test_loss=None, name=None):
        auc = self.calc_auc(all_scores, all_label).item()
        acc = self.calc_accuracy(all_scores, all_label).item()
        jsd = calc_JSD(all_scores, all_label).item()
        jsd2 = calc_JSD(all_scores).item()

        if name:
            print(f'=== {name} ===')
        if train_loss:
            print('train loss:', train_loss)
        if val_loss:
            print('val loss:', val_loss)
        if test_loss:
            print('test loss:', test_loss)
        print('accuracy:', acc)
        print('AUC:', auc)
        print('JSD:', jsd)
        print('JSD w/o labels:', jsd2)
        # print('calibration_error:', calibration_error)
        print('') 
        sys.stdout.flush()
        
def calc_JSD(ratio, label=None):
    if label is None:
        label = ratio
    d1 = torch.mean(label*torch.log2(ratio+1e-8))
    d2 = torch.mean((1-label)*torch.log2(1.-ratio+1e-8))
    return d1+d2+1.

def plotting_losses(train_loss, val_loss, test_loss, dir_name=''):  
    plt.figure(1)
    plt.plot(val_loss, label='val')
    plt.plot(test_loss, label='test')
    plt.plot(train_loss, label='train')
    plt.legend()
    plt.savefig("/data/dust/user/mmozzani/pion-clouds/figs/classifier/"+dir_name+"/loss.png")
    plt.close()
    
