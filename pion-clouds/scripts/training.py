from comet_ml import Experiment
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import math
import matplotlib.pyplot as plt
from utils.dataset import PionClouds
from models.common import get_linear_scheduler
from utils.misc import *
from models.CaloClouds_2 import CaloClouds2_Attention
import k_diffusion as K
from scipy.optimize import linear_sum_assignment
import argparse
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import wasserstein_distance
import subprocess
from adam_mini import Adam_mini

parser = argparse.ArgumentParser()
parser.add_argument('--type', default="hcal", type=str)
params = parser.parse_args()

# Load the config.yaml file
if params.type == 'hcal': config_file_path = 'configs/configs_HCAL.yaml'
elif params.type == 'ecal': config_file_path = 'configs/configs_ECAL.yaml'
else: 
    ValueError('Type not recognized. Choose between hcal and ecal')
    
cfg = Config.from_yaml(config_file_path) 
seed_all(seed = cfg.seed)

if cfg.multi_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: 
   cfg.dataloader.train_bs = int(cfg.dataloader.train_bs // torch.cuda.device_count())
   
if cfg.scheduler.lr_scheduler == 'linear':
    cfg.scheduler.lr = 1e-4 # as CC2

if cfg.log_comet:   
    with open('comet_api_key.txt', 'r') as file:
        key = file.read()
 
# Dataset and loader
train_dset = PionClouds(
    files_path = cfg.data.dataset_path,
    bs = cfg.dataloader.train_bs,
    only_hcal = cfg.only_hcal,
    only_ecal = cfg.only_ecal,
)
      
# Train, validate and test
class Train(Module):
    def __init__(self, experiment, rank):# model, model_ema, ema_sched, optimizer, scheduler):
        super().__init__()
        self.model, self.model_ema, self.ema_sched, self.it, checkpoint = self.get_model(rank)
        self.optimizer, self.scheduler = self.get_optimizer_scheduler(self.model, checkpoint) 
        self.experiment = experiment
        # Sigma (time step) distibution --> lognormal distribution, so minimum value is 0
        self.sample_density = K.config.make_sample_density(cfg.to_dict(['model_sigma'])['model_sigma']) 
    
    def get_it(self):
        return self.it
    
    def get_model(self, gpu_id):
        model = CaloClouds2_Attention(cfg)
        model_ema = CaloClouds2_Attention(cfg)

        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_parameters_ema = sum(p.numel() for p in model_ema.parameters() if p.requires_grad)
        print(num_parameters, num_parameters_ema)  
            
        if cfg.multi_gpu:
            model = DDP(model.to(gpu_id), device_ids=[gpu_id])
            model_ema = DDP(model_ema.to(gpu_id), device_ids=[gpu_id])
        else: 
            model = model.to(gpu_id)
            model_ema = model_ema.to(gpu_id)
            
        if cfg.model_path is not None:
            if len(cfg.model_path.split('/')[1].split('_')) == 2: # in this way I can say load the latest model and it finds the right iteration (it)
                it=0
                for file in os.listdir(cfg.logdir+cfg.model_path.split('/')[0]):
                    if len(file.split('_')) == 3:
                        it_temp = int(file.split('_')[2].split('.')[0]) 
                        if it_temp > it: it=it_temp            
            else: it = int(cfg.model_path.split('/')[1].split('_')[2].split('.')[0])
            cfg.scheduler.sched_start_epoch = it 
            checkpoint = torch.load(cfg.logdir+cfg.model_path, map_location=torch.device(cfg.device)) #, weights_only=True)
            
            # check wheter the check point comes from multi or single gpu
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                if not k.startswith("module.module.") and k.startswith("module."):
                    new_key = "module." + k
                else:
                    new_key = k
                state_dict[new_key] = v
            model = load_model_state(model, state_dict, multi_gpu_model=cfg.multi_gpu)
            # model = load_model_state(model, checkpoint['state_dict'], multi_gpu_model=cfg.multi_gpu)
            
            # Set the starting lr to the last checkpoint lr
            cfg.scheduler.lr = checkpoint['others']['optimizer']['param_groups'][0]['lr']    
        else: 
            it = 1
            checkpoint = None 

        # initiate EMA (exponential moving average) model
        model_ema.load_state_dict(model.state_dict())
        model_ema.eval().requires_grad_(False)
        assert cfg.ema_type == 'inverse'
        ema_sched = K.utils.EMAWarmup(power=cfg.ema_power,
                                        max_value=cfg.ema_max_value)

        return model, model_ema, ema_sched, it, checkpoint

    def get_optimizer_scheduler(self, model, checkpoint):
        if cfg.multi_gpu: diff = model.module.diffusion.parameters()
        else: diff = model.diffusion.parameters()
    
        # Optimizer and scheduler
        if cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                [
                {'params': diff},
                ], 
                lr=cfg.scheduler.lr,  
                weight_decay=cfg.scheduler.weight_decay
            )
        elif cfg.optimizer == 'RAdam':
            optimizer = torch.optim.RAdam(   # Consistency Model was trained with Rectified Adam, in k-diffusion AdamW is used, in EDM normal Adam
                    [
                    {'params': diff},
                    ], 
                    lr=cfg.scheduler.lr,  
                    weight_decay=cfg.scheduler.weight_decay
                )
        if cfg.optimizer == 'AdamMini':
            verbose_adammini = True
            if cfg.multi_gpu: 
                n = model.module.diffusion.named_parameters()
                if dist.get_rank() != 0: verbose_adammini = False
            else: n = model.diffusion.named_parameters()
            
            optimizer = Adam_mini(
                named_parameters = n,
                lr = cfg.scheduler.lr,
                betas = (0.9, 0.999), #same as Adam
                eps = 1e-08, #same as Adam
                weight_decay = cfg.scheduler.weight_decay,
                dim = cfg.transformer.d_model,
                n_heads = cfg.transformer.nhead,
                verbose=verbose_adammini
                )
            optimizer.wqk_names.add('in_proj')
            optimizer.wv_names.add('in_proj')
            optimizer.attn_proj_names.add('out_proj')
            # optimizer.output_names.add('decoder')
        
        else: 
            raise NotImplementedError('Optimizer not implemented')

        if cfg.scheduler.lr_scheduler == 'linear':
            print('Linear Scheduler as CC2')
            scheduler = get_linear_scheduler(
                    optimizer,
                    start_epoch=cfg.scheduler.sched_start_epoch, 
                    end_epoch=cfg.scheduler.sched_end_epoch, 
                    start_lr=cfg.scheduler.lr, 
                    end_lr=cfg.scheduler.end_lr 
                )
        elif cfg.scheduler.lr_scheduler == 'onecyclelr':
            print('One Cycle Scheduler')
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=cfg.scheduler.max_lr, 
                total_steps=cfg.scheduler.sched_end_epoch,
                pct_start = cfg.scheduler.pct_start, 
                div_factor = 10, # default is 25 
                final_div_factor = 200, # default is 10^4 but is too large it this case
                cycle_momentum=False,
                anneal_strategy='cos'
            )    
        elif cfg.scheduler.lr_scheduler == 'cosineannealinglr':
            print('Cosine Annealing Scheduler with Linear Warm Up')
            # this is the way has been implemented in https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                        optimizer, 
                                                        T_max = cfg.scheduler.sched_end_epoch - cfg.scheduler.sched_start_epoch, 
                                                        eta_min = cfg.scheduler.end_lr
                                                        )
            if cfg.scheduler.lr_warmup_epochs > 0: 
                # the initial lr is determined by the optimizer
                scheduler1 = torch.optim.lr_scheduler.StepLR(
                                                    optimizer, 
                                                    step_size=1, 
                                                    gamma=cfg.scheduler.lr_warmup_factor
                                                    )
                
                scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                        optimizer, 
                                                        T_max = cfg.scheduler.sched_end_epoch - cfg.scheduler.sched_start_epoch, 
                                                        eta_min = cfg.scheduler.end_lr
                                                        ) 
                # Manually override optimizer's learning rate to match warm-up final LR
                s = scheduler2.state_dict()
                s['base_lrs'] =  [((cfg.scheduler.lr_warmup_factor)**(cfg.scheduler.sched_start_epoch)) * cfg.scheduler.lr]
                scheduler2.load_state_dict(s)

                scheduler = torch.optim.lr_scheduler.SequentialLR(
                                                    optimizer, 
                                                    schedulers = [scheduler1, scheduler2],  
                                                    milestones = [cfg.scheduler.sched_start_epoch]
                                                    )  
                # bring back the optimizer to the initial lr
                for param_group in optimizer.param_groups: param_group['lr'] = cfg.scheduler.lr
                
        if checkpoint is not None: 
            optimizer.load_state_dict(checkpoint['others']['optimizer'])
            scheduler.load_state_dict(checkpoint['others']['scheduler'])        
        return optimizer, scheduler
    
    def logE_and_padding_sampled_showers(self, fake_shower, padding=None):
        if cfg.data.log_energy:
            fake_shower[:, :, 3] *= cfg.data.log_var
            fake_shower[:, :, 3] += cfg.data.log_mean
            fake_shower[:, :, 3] = np.exp(fake_shower[:, :, 3])
        
        if padding is not None:
            # removing the padded values and set them to zero
            if padding.shape == fake_shower[:,:,0].shape: 
                for i in range(4): fake_shower[:,:,i][padding] = 0
            elif padding.shape[1] > fake_shower.shape[1]:
                for i in range(4): fake_shower[:,:,i][padding[:,:fake_shower.shape[1]]] = 0 
            elif padding.shape[1] < fake_shower.shape[1]:
                for i in range(4): fake_shower[:,:padding.shape[1],i][padding] = 0 
        return fake_shower
     
    def plot_during_train(self, x2, fake_shower): 
        k=1    
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,5))
        ax1.scatter(fake_shower[k,:,0], fake_shower[k,:,1], s=fake_shower[k,:,3])
        ax1.set_ylim(-1, 1)
        ax2.scatter(x2[k,:,0], x2[k,:,1], s=x2[k,:,3])
        ax2.set_ylim(-1, 1)
        bb = np.logspace(np.log10(1e-3), np.log10(1e3), 100)
        ax3.hist(x2[:,:,3][x2[:,:,3]>0].flatten(), bins=bb, color='lightgray')
        ax3.hist(fake_shower[:,:,3][fake_shower[:,:,3]>0].flatten(), bins=bb, linewidth=3, histtype='step', color='green')
        ax3.set_yscale('log')
        ax3.set_xscale('log')  
        ax4.hist(x2[:,:,3].sum(axis=1).flatten(), bins=10, color='lightgray')
        ax4.hist(fake_shower[:,:,3].sum(axis=1).flatten(), bins=10, linewidth=3, histtype='step', color='green')
        ax4.set_yscale('log')  
        ax4.set_xlabel('energy sum')
        self.experiment.log_figure('Pion Shower ', figure=fig, step=self.it)
        plt.close()
    
    def compute_wasserstein(self, real, fake):
        if cfg.only_ecal:  # layer 29 is the max
            Ymin, Ymax = 0, 30
        elif cfg.only_hcal:  # layer 77 is the max
            Ymin, Ymax = 30, 78
        max_clip = (Ymax-1 -Ymin) /(Ymax-Ymin) *2 -1 
        
        en_sum_list, en_per_l_list, cog_x, cog_y, cog_z = [], [], [], [], []
        for s in [real, fake]:
            en_sum_list.append(s[:,:,3].sum(axis=1))  
            cog_x.append(np.sum((s[:,:,0] * s[:,:,3]), axis=1) / s[:,:,3].sum(axis=1))
            cog_y.append(np.sum((s[:,:,1] * s[:,:,3]), axis=1) / s[:,:,3].sum(axis=1))
            cog_z.append(np.sum((s[:,:,2] * s[:,:,3]), axis=1) / s[:,:,3].sum(axis=1))  
            s[:, :, 1] = np.clip(s[:, :, 1], -1, max_clip)
            s[:, :, 1] = (s[:, :, 1] + 1) / 2
            s[:, :, 1] = s[:, :, 1] * (Ymax - Ymin) + Ymin     
            en_per_l = []
            for l in np.arange(Ymin, Ymax, 1):
                cut = (s[:, :, 1]>=l) & (s[:, :, 1]<l+1)
                en_per_l.append(s[:, :, 3][cut].sum() / s.shape[0]) 
            en_per_l_list.append(np.array(en_per_l))
             
        wd_en_sum = wasserstein_distance(en_sum_list[0], en_sum_list[1])
        wd_cog_x = wasserstein_distance(cog_x[0], cog_x[1]) 
        wd_cog_y = wasserstein_distance(cog_y[0], cog_y[1]) 
        wd_cog_z = wasserstein_distance(cog_z[0], cog_z[1]) 
        wd_en_per_l = wasserstein_distance(en_per_l_list[0], en_per_l_list[1])  
        
        self.experiment.log_metric('WD Energy Sum', wd_en_sum, self.it)
        self.experiment.log_metric('WD Cog X', wd_cog_x, self.it)
        self.experiment.log_metric('WD Cog Y', wd_cog_y, self.it)
        self.experiment.log_metric('WD Cog Z', wd_cog_z, self.it)
        self.experiment.log_metric('WD Energy per Layer', wd_en_per_l, self.it)
        
    def solve_assignment(self, dist_matrix_slice):
        _, b = linear_sum_assignment(dist_matrix_slice)
        return b
    
    def assignment_problem_point_clouds(self, noise, point_cloud):
        """
        Solves the assignment problem for point clouds using the Hungarian algorithm (for OTA).
        Args:
            noise (torch.Tensor): A noisy point cloud of shape (N, M, 3), where N is the batch size,
                                M is the number of points, and 3 represents (x, y, z) coordinates.
            point_cloud (torch.Tensor): The clean point cloud of shape (N, M, 3).
        Returns:
            torch.Tensor: The reordered `noise` point cloud based on the optimal assignment.
        """
        N, M, _ = point_cloud.shape  # Batch size, number of points, 3D coordinates
        # Compute pairwise Euclidean distances (N x M x M)
        dist_matrix = torch.cdist(point_cloud, noise).detach().cpu().numpy()  # Uses L2 norm by default

        # Solve the assignment problem for each batch independently in parallel
        with ThreadPoolExecutor() as executor:
            assigned_indices = list(executor.map(self.solve_assignment, dist_matrix))

        # Convert indices back to PyTorch tensor and ensure the same device as `noise`
        assigned_indices = torch.tensor(np.array(assigned_indices), device=noise.device, dtype=torch.long)
        # Gather the reordered noise points based on the optimal assignment
        reordered_noise = torch.gather(noise, 1, assigned_indices.unsqueeze(-1).expand(-1, -1, 4))
        return reordered_noise

    def load_data_per_loop(self, batch, device):
        # Load data
        x = batch['event'][0].float() # B, N, 4
        e = batch['energy'][0].float() # B, 1
        n_tree_points = batch['n_points'][0].float().to(device) # B, 1
        points_per_layer = batch['points_per_layer'][0].float() # B, 78 (ECAL+HCAL => 78 layers) | only_hcal = 48 layers
        pm = batch['padding_mask'][0].bool().to(device) 
        points_per_layer_max = batch['points_per_layer_max'][0].float().to(device)
        if cfg.only_hcal: 
            cond_ecal = batch['cond_ecal'][0].float() # B, num_points, 4
            cond_ecal = torch.Tensor(cond_ecal).to(device)
        else: 
            cond_ecal = None
            cfg.data.ecal_compressed = False
                    
        if self.get_it() ==10: print(e)
        if cfg.data.norm_cond:
            e = e / 100 * 2 -1   # assumse max incident energy: 100 GeV
        
        cond_feats = torch.cat([e, points_per_layer], -1).to(device)
        for j in range(x.shape[0]):
            x[j,:,3][pm[j]] = 0
        
        if cfg.data.log_energy:
            x[:, :, 3] = np.log(x[:, :, 3] + 1e-20)
            x[:, :, 3] = x[:, :, 3] - cfg.data.log_mean
            x[:, :, 3] = x[:, :, 3] / cfg.data.log_var 
        
        return x, cond_feats, cond_ecal, n_tree_points, pm, points_per_layer_max 
        
    def forward(self, batch, it, device):   
        self.it = it
        cfg.device = device 
        global time_tick 
        time_tick = time.time() 
        writer = self.experiment # cfg.log_comt=False: self.experiment is None
        
        x, cond_feats, cond_ecal, n_tree_points, pm, points_per_layer_max  = self.load_data_per_loop(batch, device)
                                    
        # Reset grad and model state 
        self.optimizer.zero_grad()     
        x = x.to(device)       
                    
        noise = torch.randn_like(x).to(device) # noise for forward diffusion
        sigma = self.sample_density([x.shape[0]], device=x.device)  # time steps

        # based on https://arxiv.org/pdf/2403.05069 
        if cfg.data.optimal_transport: #new!
            noise = self.assignment_problem_point_clouds(noise, x) 
        
        if cfg.multi_gpu:
            loss = self.model.module.get_loss(x, noise, sigma, cond_feats, kl_weight=cfg.model.kl_weight, writer=writer, it=it, kld_min=cfg.model.kld_min, padding_mask= pm, cond_ecal=cond_ecal)
        else:
            loss = self.model.get_loss(x, noise, sigma, cond_feats, kl_weight=cfg.model.kl_weight, writer=writer, it=it, kld_min=cfg.model.kld_min, padding_mask= pm, cond_ecal=cond_ecal)

        if torch.isnan(loss): 
            print('Loss is NaN')
            print(np.argwhere(np.isnan(pm.cpu().numpy())))
            print(np.argwhere(np.isnan(cond_ecal.cpu().numpy())))
            print(np.argwhere(np.isnan(cond_feats.cpu().numpy())))
            print(np.argwhere(np.isnan(x.cpu().numpy())))
            print(loss.item())
            print((~pm).sum(axis=1))
            sys.exit()
    
        # Backward and optimize
        loss.backward()
        if cfg.multi_gpu: orig_grad_norm = clip_grad_norm_(self.model.module.parameters(), cfg.max_grad_norm)
        else: orig_grad_norm = clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # Update EMA model 
        ema_decay = self.ema_sched.get_value()
        K.utils.ema_update(self.model, self.model_ema, ema_decay)
        self.ema_sched.step()

        if (it % cfg.log_iter == 0) | (it == 10):
            time_elapsed = time.time() - time_tick
            print('[Train] Iter %04d | Time %.4f | Loss %.6f | Grad %.4f | KLWeight %.4f | EMAdecay %.4f' % (
                    it, time_elapsed, loss.item(), orig_grad_norm, cfg.model.kl_weight, ema_decay
            ))
            if cfg.log_comet:
                self.experiment.log_metric('train/loss', loss, it)
                self.experiment.log_metric('train/kl_weight', cfg.model.kl_weight, it)
                self.experiment.log_metric('train/lr', self.optimizer.param_groups[0]['lr'], it) 
                self.experiment.log_metric('train/grad_norm', orig_grad_norm, it)
                self.experiment.log_metric('train/ema_decay', ema_decay, it)

                if (it % (25*cfg.log_iter) == 0)| (it == 10):
                    k = 1
                    max_cluster = int((n_tree_points.cpu()*3200).max())
                    
                    if cfg.multi_gpu: 
                        fake_shower = self.model.module.sample(cond_feats, max_cluster, 
                                                               cfg, cond_ecal=cond_ecal, 
                                                               padding = None)
                        
                    else: fake_shower = self.model.sample(cond_feats, max_cluster, 
                                                cfg, cond_ecal= cond_ecal,
                                                padding = None,
                                                )
                    
                    x2 = x.detach().cpu().numpy()
                    fake_shower = fake_shower.detach().cpu().numpy()
                    padding = pm.detach().cpu().numpy()

                    num_clusters = (cond_feats[:, 1:] * points_per_layer_max).detach().cpu().numpy().sum(axis=1).astype(int)
                    padding2 = np.zeros((fake_shower.shape[0], max_cluster)).astype(bool)
                    col_indices = np.arange(max_cluster)
                    padding2[col_indices >= np.array(num_clusters.reshape(-1))[:, None]] = True 

                    fake_shower = self.logE_and_padding_sampled_showers(fake_shower, padding=padding2)
                    x2 = self.logE_and_padding_sampled_showers(x2, padding=None)
                    if cfg.log_comet:
                        self.plot_during_train(x2, fake_shower)
                        self.compute_wasserstein(x2, fake_shower)  
                    
            time_tick = time.time()
        return self.model, self.model_ema, self.ema_sched, self.optimizer, self.scheduler

# Main loop
def main(rank, world_size):
    if (cfg.model_path is not None) | (rank>0): 
        if rank>0: log_dir = None
        else: log_dir = cfg.logdir + cfg.model_path.split('/')[0]
    else:
        log_dir = get_new_log_dir(
                                cfg.logdir, 
                                prefix=cfg.name, 
                                postfix='_' + cfg.tag if cfg.tag is not None else '', 
                                start_time=time.localtime()
        )
    if rank == 0:
        ckpt_mgr = CheckpointManager(log_dir)
        bash_command = f"cp {config_file_path} {log_dir}/"
        print("copying the config file inside the checkpoint directory...")
        print(bash_command)
        result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)

    if cfg.multi_gpu:
        ddp_setup(rank, world_size)
        dataloader = DataLoader(
            train_dset,
            batch_size=1,
            num_workers=cfg.dataloader.workers,
            # shuffle=cfg.shuffle, # shuffle is done in the sampler
            sampler = DistributedSampler(train_dset),
        )
        # otherwise I have 4 running proccesses and I get 4 times the same log
        if rank!=0: cfg.log_comet=False
    else:
        dataloader = DataLoader(
            train_dset,
            batch_size=1,
            num_workers=cfg.dataloader.workers,
            shuffle=cfg.dataloader.shuffle,
        ) 

    if cfg.log_comet:
        experiment = Experiment(
            project_name=cfg.comet_project, auto_metric_logging=False, api_key=key,
        )
        experiment.log_parameters(cfg.__dict__)
        experiment.set_name(log_dir.split('/')[-1])
        if cfg.only_hcal: experiment.log_code(file_name='configs/configs_HCAL.yaml')
        elif cfg.only_ecal: experiment.log_code(file_name='configs/configs_ECAL.yaml')
        experiment.log_code(file_name='scripts/training.py')
    else: experiment = None   
    
    start_time = time.time()  
    stop = False
    it_per_one_epoch = len(dataloader) / cfg.dataloader.train_bs
    # initializing Train class
    train = Train(experiment, rank)
    it = train.get_it()
    epoch = it/int(it_per_one_epoch)
    
    if dist.is_initialized():
        dist.destroy_process_group()  # Ensure no distributed operations are running
        
    if not dist.is_initialized() or dist.get_rank() == 0: print('Start training...')
    
    for _ in range(cfg.epochs):
        #note: the batch here are changed dinamically. From the dataloader I have indeces which are 
        # the number of the input data (2M). Then every time the dataloader chooses an index, I take the showers
        # from that index to index + batch_size.
        
        for batch in dataloader:
            if it % int(it_per_one_epoch) == 0: 
                print('EPOCH: %d'%(epoch))
                epoch+=1
                
            it += 1
            model, model_ema, ema_sched, optimizer, scheduler = train(batch, it, rank)
            
            if rank==0:
                if (it % cfg.val_freq == 0 or it == cfg.max_iters) or it == 1000:
                    if cfg.multi_gpu: mod = model_ema.module.state_dict() 
                    else: mod = model_ema.state_dict() 
                    opt_states = {
                        'model_ema': mod, # save the EMA model
                        'ema_sched': ema_sched.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    
                    ckpt_mgr.save(model, cfg, 0, others=opt_states, step=it)
                    ckpt_mgr.save(model, cfg, 0, others=opt_states, save_latest=True)
                    ckpt_mgr.save(model, cfg, 0, others=opt_states, save_best=True)
                
    if cfg.multi_gpu: destroy_process_group()      
    print('training done in %.2f seconds' % (time.time() - start_time))
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size>1: cfg.multi_gpu=True
    else: cfg.multi_gpu=False
    
    if cfg.multi_gpu: mp.spawn(main, args=(world_size,), nprocs=world_size)
    else: main(world_size-1, cfg.device)
    