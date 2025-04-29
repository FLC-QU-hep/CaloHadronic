import torch
from torch.nn import Module, ModuleList, functional
import torch.nn as nn
import numpy as np
from functools import partial
# from scripts.model.common import *
from utils.misc import mean_flat
from tqdm.auto import trange, tqdm 
import k_diffusion as K 
import matplotlib as mlp 
import matplotlib.pyplot as plt 
mlp.rcParams['text.usetex'] = False

class CaloClouds2_Attention(Module):

    def __init__(self, args, distillation = False):
        super().__init__()
        self.args = args
        self.distillation = distillation

        net = PointCloudNet(
            point_dim=args.data.features, 
            context_dim= args.data.cond_features,
            ecal_dim = args.data.ecal_features,
            ecal_compressed = args.data.ecal_compressed,
            nhead=args.transformer.nhead,
            d_model=args.transformer.d_model,
            num_layers=args.transformer.num_layers,
            num_layers_dec_attn=args.transformer.num_layers_dec_attn,
            dim_feedforward=args.transformer.dim_feedforward,
            decoder_crossattn=args.transformer.decoder_crossattn,
            dropout_rate = args.transformer.dropout_rate,
            **args.transformer.to_dict(['embed_kwargs'])['embed_kwargs']
        )
        
        if distillation:
            raise ValueError("distillation should be False")
        else: 
            self.diffusion = Denoiser(net, point_dim=args.data.features, sigma_data=args.model_sigma.sigma_data, device=args.device, diffusion_loss=args.diffusion_loss, diffusion_way=args.diffusion_way)
          
            
        self.kld = KLDloss()

    def get_loss(self, x, noise, sigma, cond_feats, kl_weight, writer=None, it=None, kld_min=0.0, padding_mask=None, cond_ecal=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            noise: Noise point cloud (B, N, d).
            sigma: Time (B, ).
            cond_feats: conditioning features (B, C) 
        """
        z = cond_feats
        loss_diffusion = self.diffusion.loss(x, noise, sigma, context=z, padding_mask=padding_mask, cond_ecal=cond_ecal).mean()    # diffusion loss
        loss = loss_diffusion
        return loss

    def append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
        return x[(...,) + (None,) * dims_to_append]

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / self.append_dims(sigma, x.ndim)
    
    def sample(self, cond_feats, num_points, config, cond_ecal=None, padding=None):
        batch_size, _ = cond_feats.size()
        z = cond_feats  # B, C
        x_T = torch.randn([z.size(0), num_points, config.data.features], device=z.device) * config.sigma_max

        if not self.distillation: 
            sigmas = K.sampling.get_sigmas_karras(config.num_steps, config.sigma_min, config.sigma_max, rho=config.rho, device=z.device)
            if config.sampler == 'euler':
                x_0 = K.sampling.sample_euler(self.diffusion, x_T, sigmas, extra_args={'context' : z}, disable=True)
            elif config.sampler == 'heun': 
                x_0 = K.sampling.sample_heun(self.diffusion, x_T, sigmas, extra_args={'context' : z, 'padding_mask' : padding, 'cond_ecal': cond_ecal}, s_churn=config.s_churn, s_noise=config.s_noise, disable=True)
            elif config.sampler == 'dpmpp_2m':
                x_0 = K.sampling.sample_dpmpp_2m(self.diffusion, x_T, sigmas, extra_args={'context' : z}, disable=True)
            elif config.sampler == 'dpmpp_2s_ancestral':
                x_0 = K.sampling.sample_dpmpp_2s_ancestral(self.diffusion, x_T, sigmas, extra_args={'context' : z}, disable=True)
            elif config.sampler == 'sample_euler_ancestral':
                x_0 = K.sampling.sample_euler_ancestral(self.diffusion, x_T, sigmas, extra_args={'context' : z}, disable=True)
            elif config.sampler == 'sample_lms':
                x_0 = K.sampling.sample_lms(self.diffusion, x_T, sigmas, extra_args={'context' : z}, disable=True)
            elif config.sampler == 'sample_dpmpp_2m_sde':
                x_0 = K.sampling.sample_dpmpp_2m_sde(self.diffusion, x_T, sigmas, extra_args={'context' : z}, disable=True)
            else:
                raise NotImplementedError('Sampler not implemented')
            
        else:  # one step for consistency model
            x_0 = self.diffusion.forward(x_T, config.sigma_max, context=z)

        return x_0
    

# from: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py#L12
class Denoiser(nn.Module):
    """A Karras et al. preconditioner for denoising diffusion models."""

    def __init__(self, inner_model, point_dim, sigma_data=0.5, device='cuda', distillation = False, sigma_min = 0.002, diffusion_loss='l2', diffusion_way='EDM'):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data1 = sigma_data
        if isinstance(sigma_data, float):
            # sigma_data = [sigma_data, sigma_data, sigma_data, sigma_data]
            sigma_data = [sigma_data for _ in range(point_dim)]
        if len(sigma_data) != point_dim:
            raise ValueError(f'sigma_data must be either a float or a list of point_dim({point_dim}) floats.')
        self.sigma_data = torch.tensor(sigma_data, device=device)   # 4,
        self.distillation = distillation
        self.sigma_min = sigma_min
        self.diffusion_loss = diffusion_loss
        self.diffusion_way = diffusion_way

    def get_scalings(self, sigma):   # B,
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)   # B, 4
        sigma = K.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)  # B, 4
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        return c_skip, c_out, c_in
    
    def get_scalings_for_boundary_condition(self, sigma):   # B,   # for consistency model
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)   # B, 4
        sigma = K.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + sigma_data**2
        )   # B, 4
        c_out = (
            (sigma - self.sigma_min)
            * sigma_data
            / (sigma**2 + sigma_data**2) ** 0.5
        )   # B, 4
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5  # B, 4
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, **kwargs):
        # c_skip, c_out, c_in = [K.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]   # B,1,1
        c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma)]   # B,1,4
        
        if self.diffusion_way == "EDM":
            noised_input = input + noise * K.utils.append_dims(sigma, input.ndim)
            if torch.isnan(noised_input).any():
                raise ValueError('model output is NaN')
            model_output, mask = self.inner_model(noised_input * c_in, sigma, **kwargs)
            target = (input - c_skip * noised_input) / c_out
            
            if self.diffusion_loss == 'EDM-monotonic': 
                # from Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation
                weights = (sigma ** 2 + self.sigma_data1 ** 2) / (sigma * self.sigma_data1) ** 2
                lamda = -2 * torch.log(sigma) 
                peak_id = torch.argmax(weights) 
                edm_mono_weights = weights.clone()
                edm_mono_weights[:peak_id] = weights[peak_id] 
                
                if mask is not None:  
                    return K.utils.append_dims(edm_mono_weights, input.ndim) * (model_output[~mask] - target[~mask]).pow(2).flatten(1).mean(1)
                else: 
                    return K.utils.append_dims(edm_mono_weights, input.ndim) * (model_output - target).pow(2).flatten(1).mean(1)
            if self.diffusion_loss == 'l2':
                if mask is not None:  
                    return (model_output[~mask] - target[~mask]).pow(2).flatten(1).mean(1)
                else: 
                    return (model_output - target).pow(2).flatten(1).mean(1)
                
            elif self.diffusion_loss == 'l1':
                return (model_output - target).abs().flatten(1).mean(1)
            else:
                raise ValueError('diffusion_loss must be either l1 or l2')
        
        elif self.diffusion_way == "cosine":
            # this is wrong!!!!
            alpha_t = torch.cos((torch.pi / 2) * t)  # Clean data scale
            sigma_t = torch.sin((torch.pi / 2) * t)  # Noise scale  
            lambda_t = 0.5  
            noised_input = K.utils.append_dims(alpha_t, input.ndim) * input + K.utils.append_dims(sigma_t, input.ndim)  * noise
            target = (input - c_skip * noised_input) / c_out 
            model_output, mask = self.inner_model(noised_input * c_in, sigma, **kwargs)
            w_t = K.utils.append_dims(torch.exp(-lambda_t * t / 2), input.ndim)
            return w_t * (model_output[~mask] - target[~mask]).pow(2).flatten(1).mean(1)
                       
    def forward(self, input, sigma, **kwargs):   # same as "denoise" in KarrasDenoiser of CM code
        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = (
                torch.tensor([sigma] * input.shape[0], dtype=torch.float32)
                .to(input.device)
                .unsqueeze(1)
            )
        # c_skip, c_out, c_in = [K.utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        if not self.distillation:
            c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings(sigma)]   # B,1,4
        else:
            c_skip, c_out, c_in = [x.unsqueeze(1) for x in self.get_scalings_for_boundary_condition(sigma)]
        # CM code did an additional resacling of the time sigma for the time conditing
        
        model_output, mask = self.inner_model(input * c_in, sigma, **kwargs)
        return model_output * c_out + input * c_skip
    

class Embedder(nn.Module):
    "Implement the fourier layer function."

    def __init__(self, d_model, **kwargs): # to change
        super(Embedder, self).__init__()
        
        self.d = d_model
        self.kwargs = kwargs 
        self.periodic_fns = [getattr(torch, fn.split('.')[1]) for fn in self.kwargs['periodic_fns']]
        self.create_embedding_fn()
        # self.mlp_layer = nn.Linear(self.point_dim, 2*self.L)

    @staticmethod
    def apply_function(x, p_fn, freq):
        """Apply a periodic function with a given frequency."""
        return p_fn(x * freq)
    
    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += self.d
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0, self.kwargs['max_freq'], self.kwargs['N_freqs'])
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.kwargs['max_freq'], self.kwargs['N_freqs'])

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(partial(self.apply_function, p_fn=p_fn, freq=freq))
                # embed_fns.append(lambda x, p_fn=p_fn,
                #                 freq=freq: p_fn(x * freq))
                out_dim += self.d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        self.dim_final = len(self.embed_fns)           

    def forward(self, x):
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)


class PointCloudNet(Module):

    def __init__(self, point_dim, context_dim, ecal_dim, ecal_compressed=False, d_model=128, time_dim=64, nhead=4, num_layers=6, num_layers_dec_attn=3, max_len=6500, dim_feedforward=2048, decoder_crossattn=False, dropout_rate=0, **embed_kwargs):
        super().__init__()
        fourier_scale = 16   # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling
        self.num_layers = num_layers
        self.num_layers_dec_attn = num_layers_dec_attn
        self.d_model = d_model 
        self.ecal_dim = ecal_dim
        self.context_dim = context_dim
        self.decoder_crossattn = decoder_crossattn
        self.ecal_compressed = ecal_compressed
        
        self.timestep_embeding = nn.Sequential(
            K.layers.FourierFeatures(1, time_dim, std=fourier_scale),   # 1D Fourier features --> with register_buffer, so weights are not trained
            nn.Linear(time_dim, time_dim), # this is a trainable layer
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )
         
        if self.ecal_dim > 0:
            if self.ecal_compressed:
                self.compressor = FastPointCompressor(ecal_dim, ecal_dim) 
            
            self.context_embeding = nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, int(d_model/2))
            )
            self.context_embeding_ecal = nn.Sequential(
                nn.Linear(ecal_dim, ecal_dim),
                nn.GELU(),
                nn.Linear(ecal_dim, int(d_model/2))
            )
            self.extra_layer = nn.Linear(int(d_model/2), d_model)
        else:
            self.context_embeding = nn.Sequential(
                nn.Linear(context_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, d_model)
            )

        if embed_kwargs['activate_fourier_layer']:
            if embed_kwargs['include_input']: dim_final = 4 + 4* embed_kwargs['N_freqs'] * len(embed_kwargs['periodic_fns']) 
            else: dim_final = 4* embed_kwargs['N_freqs'] * len(embed_kwargs['periodic_fns']) 
            
            self.point_embeding = nn.Sequential(
                Embedder(point_dim, **embed_kwargs),
                nn.Linear(dim_final, d_model)
                # to substitute with :
                # nn.Linear(dim_final, int(d_model/2)),
                # nn.GELU(),
                # nn.Linear(int(d_model/2), d_model)
            )
        else:
            self.point_embeding = nn.Sequential(
                nn.Linear(point_dim, point_dim*4),
                nn.GELU(),
                nn.Linear(point_dim*4, d_model)
            )
        
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout_rate, dim_feedforward=dim_feedforward)
        self.denoiser = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        if self.decoder_crossattn:  
            transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout_rate, dim_feedforward=dim_feedforward)
            self.trasf_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=1) 
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, int(d_model/2)),
            nn.GELU(),
            nn.Linear(int(d_model/2), int(d_model/4)),
            nn.GELU(),
            nn.Linear(int(d_model/4), point_dim)
        )
    
    def forward(self, x, sigma, context, padding_mask=None, study_attn=False, cond_ecal=None):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, F). 
        """
         
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)
        
        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.timestep_embeding(c_noise)  # (B, 1, T)
        ctx_emb = self.context_embeding(context)    # (B, 1, F+T)
        points_emb = self.point_embeding(x) 
                    
        if self.ecal_dim > 0: 
            if self.ecal_compressed:
                cond_ecal = self.compressor(cond_ecal)  # (B, num_points, d) --> (B, 10, d)
            ctx_ecal = self.context_embeding_ecal(cond_ecal)    # (B, num_points, d_model/2) 
            ctx_emb = torch.cat([ctx_emb, ctx_ecal], dim=1) # (B, num_points+1, d_model/2) 
            ctx_emb = self.extra_layer(ctx_emb) # (B, num_points+1, d_model) 

        x = torch.cat([points_emb, time_emb, ctx_emb], dim=1) # (B, tokens, d_model)
         
        if padding_mask is not None:  # positions with True are not allowed to attend while False values will be unchanged
            if self.ecal_dim > 0: context_pad = torch.Tensor(np.zeros(2+cond_ecal.shape[1])).bool().repeat(batch_size, 1).to('cuda') 
            else: context_pad = torch.Tensor([0,0]).bool().repeat(batch_size, 1).to('cuda') 
            mask = torch.cat([padding_mask, context_pad], dim=-1) 
        else: mask = None 
        
        x = self.denoiser(x, src_key_padding_mask=mask)        
        if self.decoder_crossattn:
            x = self.trasf_decoder(tgt=points_emb, memory=x, tgt_is_causal=False, memory_key_padding_mask=mask, tgt_key_padding_mask=padding_mask)     
        else:    
            if self.ecal_dim > 0: tokens_to_remove = 2+cond_ecal.shape[1] # 1 for time, 1 for context, num_points_ecal for ecal_cond
            else: tokens_to_remove = 2 # 1 for time, 1 for context
            x = x[:, :-tokens_to_remove] # remove last tokens
                  
        x = self.decoder(x)    
        return x, padding_mask  
    

class KLDloss(nn.Module):
    
    def __init__(self):
        super(KLDloss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        
    def forward(self, mu, logvar):
            B = logvar.size(0)
            KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/(B)
            return KLD
        
class FastPointCompressor(nn.Module):
    def __init__(self, in_dim, out_dim, num_tokens=10):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_proj = nn.Linear(in_dim, num_tokens)  # For soft assignment
        self.point_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):  # x: [B, N, in_dim]
        B, N, _ = x.shape
        assignment = torch.softmax(self.token_proj(x), dim=2)  # [B, N, K]
        x_proj = self.point_proj(x)  # [B, N, out_dim]
        # Weighted mean for each token 
        assignment = assignment.transpose(1, 2)  # [B, K, N]
        compressed = torch.bmm(assignment, x_proj) / (assignment.sum(dim=2, keepdim=True) + 1e-6)  # [B, K, out_dim]
        return compressed
