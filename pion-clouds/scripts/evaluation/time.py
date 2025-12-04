#!/usr/bin/env python
# coding: utf-8
import time
import numpy as np
import torch
import sys
import os
# Add the parent directory of 'utils' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

if sys.argv[1].lower() == "cpu":
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    tag = "cpu"
    device_name = "cpu"
else:
    torch.set_default_device("cuda")
    tag = "gpu"
    device_name = torch.cuda.get_device_name(0)

device = torch.get_default_device()

if sys.argv[2].lower() == "raw":
    tag += "_raw"
    from evaluation import minimal_gen
    cfg_ecal = minimal_gen.cfg_ecal
    cfg_ecal.device = str(device)
    cfg_hcal = minimal_gen.cfg_hcal
    cfg_hcal.device = str(device)
    edm_dir_ecal = minimal_gen.edm_dir_ecal
    edm_dir = minimal_gen.edm_dir

    model, model_ecal, distribution, distribution_ecal, cfg_hcal, cfg_ecal = minimal_gen.main(
        cfg_ecal, cfg_hcal, edm_dir_ecal, edm_dir
    )
    def run(energy, batch_size):
        fake_showers = minimal_gen.generate(
            model,
            model_ecal,
            distribution,
            distribution_ecal,
            cfg_hcal,
            cfg_ecal,
            energy,
            batch_size,
            batch_size,
        )
        return fake_showers

elif sys.argv[2].lower() == "compile":
    tag += "_compile"
    from evaluation import minimal_gen_compiled as minimal_gen
    cfg_ecal = minimal_gen.cfg_ecal
    cfg_ecal.device = str(device)
    cfg_hcal = minimal_gen.cfg_hcal
    cfg_hcal.device = str(device)
    edm_dir_ecal = minimal_gen.edm_dir_ecal
    edm_dir = minimal_gen.edm_dir

    model, model_ecal, distribution, distribution_ecal, cfg_hcal, cfg_ecal = minimal_gen.main(
        cfg_ecal, cfg_hcal, edm_dir_ecal, edm_dir
    )
    def run(energy, batch_size):
        fake_showers = minimal_gen.generate(
            model,
            model_ecal,
            distribution,
            distribution_ecal,
            cfg_hcal,
            cfg_ecal,
            energy,
            batch_size,
            batch_size,
        )
        return fake_showers

else:
    tag = "_torchDynamic"
    from evaluation import minimal_gen
    cfg_ecal = minimal_gen.cfg_ecal
    cfg_ecal.device = str(device)
    cfg_hcal = minimal_gen.cfg_hcal
    cfg_hcal.device = str(device)
    edm_dir_ecal = minimal_gen.edm_dir_ecal
    edm_dir = minimal_gen.edm_dir

    model, model_ecal, distribution, distribution_ecal, cfg_hcal, cfg_ecal = minimal_gen.main(
        cfg_ecal, cfg_hcal, edm_dir_ecal, edm_dir
    )

    @torch.compile(dynamic=True)
    def run(energy, batch_size):
        fake_showers = minimal_gen.generate(
            model,
            model_ecal,
            distribution,
            distribution_ecal,
            cfg_hcal,
            cfg_ecal,
            energy,
            batch_size,
            batch_size,
        )
        return fake_showers

if len(sys.argv) > 3:
    tag += "_" + sys.argv[3]

save_name = f"ch_timing_{tag}.npz"
print(f"Using device: {device}")
print(f"saving to {save_name}")

# Use for cpu --constraint=EPYC&7402&512G
# Use for gpu --constraint=A100-PCIE-80GB&GPUx1

# local frame of ref

energies = [10.0 * i for i in range(1, 10)]
if device == "cpu":
    batch_sizes = [1, 16, 64, 128, 256, 512]
else:
    batch_sizes = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096]

n_goes = 100
warm_up = 2
total_runs = n_goes + warm_up
times = np.zeros((len(energies), len(batch_sizes), total_runs))


for bi, batch_size in enumerate(batch_sizes):
    print(f"Batch size: {batch_size}")
    for ei, energy in enumerate(energies):
        energy = torch.FloatTensor(batch_size, 1).fill_(energy).to(device)
        if cfg_hcal.data.norm_cond and cfg_ecal.data.norm_cond:
            energy = energy / 100 * 2 - 1
        else:
            raise ValueError("cond_E Normalization not consistent between ECAL and HCAL models")
        # no warm up, because these take forever
        for i in range(total_runs):
            print(i)
            start = time.time()
            run(energy, batch_size)
            end = time.time()
            times[ei, bi, i] = (end - start) / batch_size
            print(
                f"energy: {energy.detach().cpu().numpy()[0]} GeV, repetition {i:2d}: time per shower: {times[ei, bi, i]:>6.2f} s"
            )
    np.savez(
        save_name,
        batch_sizes=batch_sizes[: bi + 1],
        times=times[:, : bi + 1],
        energies=energies,
        device_name=device_name,
    )
trimmed_times = times[:, :, warm_up:]
print(f"Mean time: {np.mean(trimmed_times)} +- {np.std(trimmed_times) / np.sqrt(len(trimmed_times))} s")
print(f"saving to {save_name}")
np.savez(
    save_name,
    batch_sizes=batch_sizes,
    times=times,
    energies=energies,
    device_name=device_name,
)
