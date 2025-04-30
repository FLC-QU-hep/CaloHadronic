import torch
import time
import sys
import os
# Add the parent directory of 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import utils.gen_utils as gen_utils
import h5py
from utils.misc import Config, Configs
from utils.misc import *
from models.CaloClouds_2 import CaloClouds2_Attention
import matplotlib.pyplot as plt
from tqdm import tqdm
import models.flow_matching as fm
from omegaconf import OmegaConf
import pickle

# min and max energy of the generated events
energy_range = [10, 90] #[84, 86] #[49, 51] #[19, 21]  #[14,16] 
num = 50002 # total number of generated events 
bs = 64 # batch size   # optimized: bs=64(cm), 64(edm), 64(ddpm) for GPU, bs=1 for CPU (single-threaded)
_with_sf = True 
generate_both_ecal_and_hcal = True
for_reco_on_pions = False 
print('for reco on pions: ', for_reco_on_pions)

caloclouds = 'edm'   # 'ddpm, 'edm', 'cm'
my_dir = '../pion-clouds-log-dir/' 
edm_dir_ecal = my_dir+'HGx9_Ecal_Smear_l3_d256_L32_CosAnn_AdamMini_Monotonic2_2025_04_08__20_07_01/'
edm_dir = my_dir+'HGx9_Hcal_ecalCompression_2025_04_11__11_30_52/'

cfg_flow = OmegaConf.load('configs/configs_sf.yaml')
cfg_hcal = Config.from_yaml(edm_dir+'configs_HCAL.yaml') 
cfg_ecal = Config.from_yaml(edm_dir_ecal+'configs_ECAL.yaml') 
configs = cfg_hcal
configs.num_input_flow = cfg_flow.fm.num_inputs
cfg_ecal.data.ecal_compressed = False
cfg_hcal.device = cfg_ecal.device = 'cuda'  # 'cuda' or 'cpu'  

for cfg in [cfg_hcal, cfg_ecal]:
    cfg.num_steps  = 30
    
gen_folder = 'HGx9_CNF_ECAL_HCAL_'+str(cfg_hcal.num_steps)  
print(gen_folder)
# use single thread
# torch.set_num_threads(1) # also comment out os.environ['OPENBLAS_NUM_THREADS'] = '1' above for multi threaded

print('num', num, 'bs', bs)     
print('steps: ', cfg_hcal.num_steps)
print('incident energy range: ', energy_range)

def main(cfg_ecal, cfg_hcal):
    flow = distribution = fm.CNF(fm.FullyConnected(**cfg_flow.fm))
    distribution_ecal = distribution 
    b = '../shower_flow_ckps/'
    checkpoint = torch.load(b+'shw_log_dir_HGx9_CNF_log_Adam-minis/ShowerFlow_bestLoss.pth', 
                            map_location=torch.device(cfg_flow.device))
    flow.load_state_dict(checkpoint['model'])
    flow.eval().to(cfg_hcal.device)
    
    model = CaloClouds2_Attention(cfg_hcal).to(cfg_hcal.device)
    model_ecal = CaloClouds2_Attention(cfg_ecal).to(cfg_ecal.device) 
    
    # this two now do not work with the new configs files! I should load the new trainings!
    checkpoint_ecal = torch.load(edm_dir_ecal+'ckpt_latest.pt', map_location=torch.device(cfg_ecal.device))
    checkpoint = torch.load(edm_dir+'ckpt_latest.pt', map_location=torch.device(cfg_hcal.device))
    
    model.load_state_dict(checkpoint['others']['model_ema'])
    model_ecal.load_state_dict(checkpoint_ecal['others']['model_ema'])

    model.eval()
    model_ecal.eval()
    return model, model_ecal, distribution, distribution_ecal, cfg_hcal, cfg_ecal

model, model_ecal, distribution, distribution_ecal, cfg_hcal, cfg_ecal = main(cfg_ecal, cfg_hcal, num)

print('starting gen_utils...')
cond_E = torch.FloatTensor(num, 1).uniform_(energy_range[0], energy_range[1]) 

if for_reco_on_pions:
    peter_en = np.load("../Pion_50GeV_Calo_Entry_Energies.npy") / 1e3
    cond_E = torch.Tensor(peter_en).unsqueeze(1)

if cfg_hcal.data.norm_cond and cfg_ecal.data.norm_cond: 
    cond_E = cond_E/ 100 * 2 -1
else: 
    raise ValueError('cond_E Normalization not consistent between ECAL and HCAL models')

 
fake_showers_ecal, samples, cond_E, real_showers, cond_E_real = gen_utils.gen_showers_batch(model_ecal, distribution_ecal, 
                                                                energy_range[0], energy_range[1], num=num, max_points=3200, 
                                                                bs=bs, kdiffusion=kdiffusion, config=cfg_ecal, 
                                                                enable_shower_flow = _with_sf, cond_E=cond_E, 
                                                                single_SF=1, gen_both_EandHcal=generate_both_ecal_and_hcal)

if cfg_hcal.device == 'cuda': torch.cuda.empty_cache()
print('generating HCAL part...')
fake_showers, _, cond_E, _, _ = gen_utils.gen_showers_batch(model, distribution, energy_range[0], energy_range[1], num=num, 
                                                   max_points=3200, bs=bs, kdiffusion=kdiffusion, config=cfg_hcal, 
                                                   enable_shower_flow = _with_sf, cond_E=cond_E, single_SF=samples, 
                                                   cond_ECAL=fake_showers_ecal, gen_both_EandHcal=generate_both_ecal_and_hcal,
                                                   real_showers=real_showers, cond_E_real=cond_E_real)
if cfg_hcal.device == 'cuda': torch.cuda.empty_cache()
n_hits_max = int(fake_showers.shape[2])
 
print('saving files...')
e_min, e_max = energy_range[0], energy_range[1]
if e_max<18: inc_en = '15GeV_'
elif (e_max<24) & (e_min>15): inc_en = '20GeV_' 
elif (e_max<54) & (e_min>45): inc_en = '50GeV_'
elif (e_max<89) & (e_min>80): inc_en = '85GeV_'
else: inc_en = '' #default

os.makedirs("../files/generated_showers/"+gen_folder, exist_ok=True)
float_type = "f8"

if for_reco_on_pions: # this was added when Anatolii asked me the showers for running reco
    file = h5py.File("/data/dust/user/mmozzani/public_repo/gen_showers_PeterEn_"+inc_en+str(num)+".hdf5", "w")
    for i in range(4): fake_showers[:,i][fake_showers[:,3]<=0] = 0
    fake_showers = np.moveaxis(fake_showers, -1, -2)
    print(fake_showers.shape)
    events_goup = file.create_dataset("events", (num, n_hits_max, 4), maxshape = (None, n_hits_max, 4), dtype = float_type)   
else: 
    file = h5py.File("/data/dust/user/mmozzani/pion-clouds/files/generated_showers/"+gen_folder+"/gen_showers_"+inc_en+str(num)+".hdf5", "w")
    events_goup = file.create_dataset("events", (num, 4, n_hits_max), maxshape = (None, 4, n_hits_max), dtype = float_type)

print(file, fake_showers.shape)
energy_goup = file.create_dataset("energy", (num, 1), maxshape = (None, 1), dtype = float_type)
file["events"].write_direct(np.ascontiguousarray(fake_showers))    #, dtype='f8'
file["energy"].write_direct(np.ascontiguousarray(cond_E))    #, dtype='f8'
file.close()
