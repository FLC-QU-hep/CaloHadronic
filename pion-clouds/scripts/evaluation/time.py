import torch
import sys
import os
import numpy as np

# Add the parent directory of 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import utils.gen_utils as gen_utils
from utils.misc import Config
from models.CaloClouds_2 import CaloClouds2_Attention
import models.flow_matching as fm
from omegaconf import OmegaConf


base_dir = "/data/dust/user/dayhallh/calohadronic-data/"


def prep_batch(evt_id, bs, hits_per_layer_all_tot, cond_E, max_points, device):
    cond_E_batch = cond_E[evt_id : evt_id + bs]
    hits_per_layer_all = hits_per_layer_all_tot[evt_id : evt_id + bs]
    num_clusters = np.sum(hits_per_layer_all, axis=1).reshape(bs, 1)  # B,1
    hits_per_layer_all[hits_per_layer_all < 0] = 0

    cond_N = (
        torch.Tensor(hits_per_layer_all.sum(axis=1) / max_points)
        .to(device)
        .unsqueeze(-1)
    )
    max_num_clusters = hits_per_layer_all.sum(axis=1).max()
    padding_mask = np.zeros((bs, max_num_clusters)).astype(bool)
    col_indices = np.arange(max_num_clusters)
    padding_mask[col_indices >= np.array(num_clusters.reshape(-1))[:, None]] = True
    cond_Nperlayer = torch.Tensor(hits_per_layer_all / hits_per_layer_all.max()).to(
        device
    )
    return cond_E_batch, cond_N, cond_Nperlayer, padding_mask, max_num_clusters


def linear_e_rescales(fs, padding_mask, Ymax, Ymin, max_clip, config):
    assert not config.data.log_energy

    fs = fs.cpu().numpy()
    fs[:, :, 1] = np.clip(fs[:, :, 1], -1, max_clip)
    fs[:, :, 1] = (fs[:, :, 1] + 1) / 2
    fs[:, :, 1] = fs[:, :, 1] * (Ymax - Ymin) + Ymin
    for i in range(4):
        fs[:, :, i][padding_mask] = 0
    return fs


def log_e_rescales(fs, padding_mask, Ymax, Ymin, max_clip, config):
    assert config.data.log_energy

    fs[:, :, 3] *= config.data.log_var
    fs[:, :, 3] += config.data.log_mean
    fs[:, :, 3] = torch.exp(fs[:, :, 3])

    fs = fs.cpu().numpy()
    fs[:, :, 1] = np.clip(fs[:, :, 1], -1, max_clip)
    fs[:, :, 1] = (fs[:, :, 1] + 1) / 2
    fs[:, :, 1] = fs[:, :, 1] * (Ymax - Ymin) + Ymin
    for i in range(4):
        fs[:, :, i][padding_mask] = 0
    return fs


def implement_z_flow(
    fs, hits_per_layer_all, y_positions, padding_mask, max_num_clusters
):
    for i, hits_per_layer in enumerate(hits_per_layer_all):
        hits_per_layer[hits_per_layer < 0] = 0
        n_hits_to_concat = max_num_clusters - hits_per_layer.sum()
        z_flow = np.repeat(y_positions, hits_per_layer)

        n_hits_to_concat = max(n_hits_to_concat, 0)
        z_flow = z_flow[:max_num_clusters]

        z_flow = np.concatenate([z_flow, np.zeros(n_hits_to_concat)])
        mask = padding_mask[
            i, :
        ]  # note!--> this is already inverted because of attention
        fs[i, :, 1][mask] = 100

        idx_dm = np.argsort(fs[i, :, 1])
        fs[i, :, :] = fs[i, :, :][idx_dm]
        fs[i, :, 1] = z_flow

        for f in range(4):
            fs[i, :, f][z_flow == 0] = 0
        fs[fs[:, :, 3] <= 0] = 0  # setting events with negative energy to zero


def ecal_gen_showers_batch(
    model,
    shower_flow,
    e_min,
    e_max,
    theta=0,
    num=2000,
    bs=8,
    kdiffusion=False,
    config=None,
    max_points=5131,
    cond_E=None,
    cond_ECAL=None,
    single_SF=None,
):
    # name samples #hgx9 --> max points = 4048

    Xmax, Xmin, Zmax, Zmin, Ymin, Ymax = 450, -450, 450, -450, 0, 78

    Ymax = 30
    max_clip = 29 / 30 * 2 - 1  # layer 29 is the max

    cond_E = cond_E.to(config.device)
    # the incident energy for the SF is initialized just diving for the max
    en_max_sf = 90.1395492553711
    context = ((cond_E + 1) / 2 * 100) / en_max_sf
    p_per_l_max = 6.400257445308821  # in log scale!
    # sample from shower flow

    with torch.no_grad():
        samples = (
            shower_flow.sample((context.shape[0], 78), condition=context)
            .cpu()
            .numpy()
        )

    samples = np.exp(samples * p_per_l_max).astype(int)

    # this is a control on the number of points per layer, and the total number of points
    simaltanious_changes = 12
    while (len(np.where(samples.sum(axis=1) > 5001)[0]) > 0) | (
        len(np.where(samples > 602)[0]) > 0
    ):
        indexes = np.unique(
            np.concatenate(
                [
                    np.where(samples.sum(axis=1) > 5001)[0],
                    np.where(samples > 602)[0],
                ]
            )
        )
        for i in indexes[:simaltanious_changes]:
            with torch.no_grad():
                samples[i] = (
                    shower_flow.sample((1, 78), condition=context[i].reshape(1, 1))
                    .cpu()
                    .numpy()
                )
                samples[i] = np.exp(samples[i] * p_per_l_max).astype(int)
                # samples[i] = np.clip(samples[i], a_min=0, a_max=602)
        simaltanious_changes = 1
    samples = samples.astype(int)
    hits_per_layer_all_tot = samples[:, :30]

    mmaxx = samples.sum(axis=1).max()
    max_shower_length = mmaxx + 10
    fake_showers = np.zeros((num, 4, max_shower_length))
    # ordering per number of hits
    idx_hits_sorted = np.argsort(hits_per_layer_all_tot.sum(axis=1))
    hits_per_layer_all_tot = hits_per_layer_all_tot[idx_hits_sorted]
    cond_E = cond_E[idx_hits_sorted]
    samples = samples[idx_hits_sorted]

    for evt_id in range(0, num, bs):
        print(evt_id)
        if (num - evt_id) < bs:
            bs = num - evt_id

        cond_E_batch, cond_N, cond_Nperlayer, padding_mask, max_num_clusters = (
            prep_batch(
                evt_id, bs, hits_per_layer_all_tot, cond_E, max_points, config.device
            )
        )

        cond_ecal = None

        # generationnnnnnn
        fs = gen_utils.get_shower(
            model,
            max_num_clusters,
            cond_E_batch,
            cond_N=cond_N,
            cond_Nperlayer=cond_Nperlayer,
            cond_ecal=cond_ecal,
            theta=0,
            bs=bs,
            kdiffusion=kdiffusion,
            pm=padding_mask,
            config=config,
        )

        # Change if e is set to linear in configs
        fs = log_e_rescales(fs, padding_mask, Ymax, Ymin, max_clip, config)

        for i in range(4):
            fs[:, :, i][padding_mask] = -10  # for SF calibration
        Ymin, Ymax = 0, 30
        hits_per_layer_all = samples[evt_id : evt_id + bs][:, :30]  # [:, :Ymax]

        cell_thickness = 1
        layer_bottom_pos = np.linspace(Ymin, Ymax - 1, Ymax - Ymin)
        y_positions = layer_bottom_pos + cell_thickness / 2

        implement_z_flow(
            fs, hits_per_layer_all, y_positions, padding_mask, max_num_clusters
        )

        fs = gen_utils.x_z_shift(fs, Xmin, Xmax, Zmin, Zmax)
        fake_showers[evt_id : evt_id + bs, :, : fs.shape[2]] = fs[
            :, :, :max_shower_length
        ]

    fake_showers[:, 1, :] = fake_showers[:, 1, :] - 0.5

    return (
        fake_showers,
        samples,
        cond_E.detach().cpu().numpy().astype("float32"),
    )  # (bs, 4, num_points), (bs, 1)


def hcal_gen_showers_batch(
    model,
    shower_flow,
    e_min,
    e_max,
    theta=0,
    num=2000,
    bs=8,
    kdiffusion=False,
    config=None,
    max_points=5131,
    cond_E=None,
    cond_ECAL=None,
    single_SF=None,
):
    # name samples #hgx9 --> max points = 4048

    Xmax, Xmin, Zmax, Zmin, Ymin, Ymax = 450, -450, 450, -450, 0, 78

    Ymin = 30
    max_clip = (77 - 30) / 48 * 2 - 1  # layer 77 is the max
    tot_cond_ecal = np.moveaxis(cond_ECAL, -1, -2)
    cond_E = torch.Tensor(cond_E)
    tot_cond_ecal[:, :, 0] = (tot_cond_ecal[:, :, 0] - Xmin) * 2 / (
        Xmax - Xmin
    ) - 1  # x normalization
    tot_cond_ecal[:, :, 2] = (tot_cond_ecal[:, :, 2] - Zmin) * 2 / (
        Zmax - Zmin
    ) - 1  # z normalization
    samples = single_SF.copy()
    hits_per_layer_all_tot = samples[:, 30:]


    mmaxx = samples.sum(axis=1).max()
    max_shower_length = mmaxx + 10
    fake_showers = np.zeros((num, 4, max_shower_length))
    # ordering per number of hits
    idx_hits_sorted = np.argsort(hits_per_layer_all_tot.sum(axis=1))
    hits_per_layer_all_tot = hits_per_layer_all_tot[idx_hits_sorted]
    cond_E = cond_E[idx_hits_sorted]
    samples = samples[idx_hits_sorted]
    tot_cond_ecal = tot_cond_ecal[idx_hits_sorted]
    cond_E = torch.Tensor(cond_E).to(config.device)
    points_ECAL = (
        torch.Tensor(np.sum(tot_cond_ecal[:, :, 3] > 0, axis=1))
        .reshape(num, 1)
        .to(config.device)
    )

    for evt_id in range(0, num, bs):
        print(evt_id)
        if (num - evt_id) < bs:
            bs = num - evt_id

        cond_E_batch, cond_N, cond_Nperlayer, padding_mask, max_num_clusters = (
            prep_batch(
                evt_id, bs, hits_per_layer_all_tot, cond_E, max_points, config.device
            )
        )

        cond_ecal = tot_cond_ecal[evt_id : evt_id + bs]
        # preprocessing ecal data
        # cond_ecal[cond_ecal[:, :, 3] == 0] = 0
        _, cond_ecal, _, padding_mask_ecal = gen_utils.get_only_hcal_data(cond_ecal)
        cond_ecal_tosave = cond_ecal.copy()
        smearing = np.random.uniform(-0.49, 0.49, size=cond_ecal[:, :, 1].shape)
        cond_ecal[:, :, 1] = cond_ecal[:, :, 1] + smearing
        cond_ecal[:, :, 1] = (cond_ecal[:, :, 1] - 0) * 2 / (
            30 - 0
        ) - 1  # y normalization
        max_point_ecal = points_ECAL[evt_id : evt_id + bs].max().reshape(-1)
        cond_ecal = torch.Tensor(cond_ecal).to(config.device)

        # generationnnnnnn
        fs = gen_utils.get_shower(
            model,
            max_num_clusters,
            cond_E_batch,
            cond_N=cond_N,
            cond_Nperlayer=cond_Nperlayer,
            cond_ecal=cond_ecal,
            theta=0,
            bs=bs,
            kdiffusion=kdiffusion,
            pm=padding_mask,
            config=config,
        )

        # Change if e is set to linear in configs
        fs = log_e_rescales(fs, padding_mask, Ymax, Ymin, max_clip, config)

        cond_ecal = cond_ecal_tosave
        padding_mask = np.concatenate(
            (padding_mask_ecal, padding_mask), axis=1
        ).astype(bool)
        fs = np.concatenate((cond_ecal, fs), axis=1)
        max_num_clusters = int(max_num_clusters + max_point_ecal)
        for i in range(4):
            fs[:, :, i][padding_mask] = -10  # for SF calibration

        Ymin, Ymax = 0, 78
        hits_per_layer_all = samples[evt_id : evt_id + bs]  # [:, :Ymax]

        cell_thickness = 1
        layer_bottom_pos = np.linspace(Ymin, Ymax - 1, Ymax - Ymin)
        y_positions = layer_bottom_pos + cell_thickness / 2

        implement_z_flow(
            fs, hits_per_layer_all, y_positions, padding_mask, max_num_clusters
        )

        fs = gen_utils.x_z_shift(fs, Xmin, Xmax, Zmin, Zmax)
        fake_showers[evt_id : evt_id + bs, :, : fs.shape[2]] = fs[
            :, :, :max_shower_length
        ]

    fake_showers[:, 1, :] = fake_showers[:, 1, :] - 0.5

    return (
        fake_showers,
        samples,
        cond_E.detach().cpu().numpy().astype("float32"),
    )  # (bs, 4, num_points), (bs, 1)


def get_configs():
    conf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../configs"))
    configs_sf_path = os.path.join(conf_dir, "configs_sf.yaml")
    cfg_flow = OmegaConf.load(configs_sf_path)
    # cfg_hcal = Config.from_yaml(edm_dir+'configs_HCAL.yaml')
    # cfg_ecal = Config.from_yaml(edm_dir_ecal+'configs_ECAL.yaml')
    cfg_hcal = Config.from_yaml(os.path.join(conf_dir, "configs_HCAL.yaml"))
    cfg_ecal = Config.from_yaml(os.path.join(conf_dir, "configs_ECAL.yaml"))
    configs = cfg_hcal
    configs.num_input_flow = cfg_flow.fm.num_inputs
    cfg_ecal.data.ecal_compressed = False
    cfg_hcal.device = cfg_ecal.device = "cuda"  # 'cuda' or 'cpu'

    for cfg in [cfg_hcal, cfg_ecal]:
        cfg.num_steps = 30

    return cfg_flow, cfg_hcal, cfg_ecal


def main(cfg_ecal, cfg_hcal, edm_dir_ecal, edm_dir):
    flow = distribution = fm.CNF(fm.FullyConnected(**cfg_flow.fm))
    distribution_ecal = distribution
    showerflow_ckpt_file = os.path.join(
        base_dir,
        "shower_flow_ckps",
        "shw_log_dir_HGx9_PointsFM/ShowerFlow_bestLoss.pth",
    )
    checkpoint = torch.load(
        showerflow_ckpt_file, map_location=torch.device(cfg_flow.device)
    )
    flow.load_state_dict(checkpoint["model"])
    flow.eval().to(cfg_hcal.device)

    model = CaloClouds2_Attention(cfg_hcal).to(cfg_hcal.device)
    model_ecal = CaloClouds2_Attention(cfg_ecal).to(cfg_ecal.device)

    # this two now do not work with the new configs files! I should load the new trainings!
    checkpoint_ecal = torch.load(
        os.path.join(edm_dir_ecal, "ckpt_latest.pt"),
        map_location=torch.device(cfg_ecal.device),
    )
    checkpoint = torch.load(
        os.path.join(edm_dir, "ckpt_latest.pt"),
        map_location=torch.device(cfg_hcal.device),
    )

    model.load_state_dict(checkpoint["others"]["model_ema"])
    model_ecal.load_state_dict(checkpoint_ecal["others"]["model_ema"])

    model.eval()
    model_ecal.eval()
    return model, model_ecal, distribution, distribution_ecal, cfg_hcal, cfg_ecal


def generate(
    model,
    model_ecal,
    distribution,
    distribution_ecal,
    cfg_hcal,
    cfg_ecal,
    cond_E,
    num,
    bs,
):
    kdiffusion = True

    (
        fake_showers_ecal,
        samples,
        cond_E,
    ) = ecal_gen_showers_batch(
        model_ecal,
        distribution_ecal,
        energy_range[0],
        energy_range[1],
        num=num,
        max_points=3200,
        bs=bs,
        kdiffusion=kdiffusion,
        config=cfg_ecal,
        cond_E=cond_E,
        single_SF=1,
    )

    # if cfg_hcal.device == "cuda":
    #    torch.cuda.empty_cache()
    fake_showers, _, cond_E = hcal_gen_showers_batch(
        model,
        distribution,
        energy_range[0],
        energy_range[1],
        num=num,
        max_points=3200,
        bs=bs,
        kdiffusion=kdiffusion,
        config=cfg_hcal,
        cond_E=cond_E,
        single_SF=samples,
        cond_ECAL=fake_showers_ecal,
    )
    # if cfg_hcal.device == "cuda":
    #    torch.cuda.empty_cache()

    return fake_showers


# min and max energy of the generated events
energy_range = [10, 90]  # [84, 86] #[49, 51] #[19, 21]  #[14,16]
# num = 50002 # total number of generated events
num = 128  # total number of generated events
bs = 64  # batch size   # optimized: bs=64(cm), 64(edm), 64(ddpm) for GPU, bs=1 for CPU (single-threaded)

edm_dir_ecal = os.path.join(
    base_dir,
    "logdir_ECAL",
    "HGx9_Ecal_Smear_l3_d256_L32_CosAnn_AdamMini_Monotonic2_2025_12_03__15_43_44",
)
edm_dir = os.path.join(
    base_dir, "logdir_HCAL", "HGx9_Hcal_ecalCompression_2025_12_03__15_53_59"
)


cfg_flow, cfg_hcal, cfg_ecal = get_configs()

# use single thread
# torch.set_num_threads(1) # also comment out os.environ['OPENBLAS_NUM_THREADS'] = '1' above for multi threaded

print("num", num, "bs", bs)
print("steps: ", cfg_hcal.num_steps)
print("incident energy range: ", energy_range)


model, model_ecal, distribution, distribution_ecal, cfg_hcal, cfg_ecal = main(
    cfg_ecal, cfg_hcal, edm_dir_ecal, edm_dir
)

print("starting gen_utils...")
cond_E = torch.FloatTensor(num, 1).uniform_(energy_range[0], energy_range[1])

if cfg_hcal.data.norm_cond and cfg_ecal.data.norm_cond:
    cond_E = cond_E / 100 * 2 - 1
else:
    raise ValueError("cond_E Normalization not consistent between ECAL and HCAL models")


fake_showers = generate(
    model,
    model_ecal,
    distribution,
    distribution_ecal,
    cfg_hcal,
    cfg_ecal,
    cond_E,
    num,
    bs,
)
print("done")
