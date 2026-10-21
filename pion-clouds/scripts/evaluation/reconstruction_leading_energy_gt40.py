import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.close()
mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25
mpl.rcParams["font.size"] = 35
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "serif"

reco_gen = h5py.File("/data/dust/user/bussthor/public/pion_clouds.h5", "r")
reco_sim = h5py.File("/data/dust/user/bussthor/public/pion_clouds.h5", "r")
dict_pid = {"e-": 11, "mu-": 13, "pi+": 211, "K+": 321, "p": 2212}
pid_to_index = {pid: i for i, pid in enumerate(dict_pid.values())}
dict_pid_labels = {k: pid_to_index.get(v, -1) for k, v in dict_pid.items()}

a, b = 2, 3
bins_ = 30
idx = 0

# Base mask + require leading PFO energy > 40 GeV
mask_gen = (
    (reco_gen["energy"][:, idx] > 0)
    & (reco_gen["pid"][:, idx] > 0)
    & (reco_gen["momentum"][:, idx, 1] > -1)
    & (reco_gen["energy"][:, idx] < 40)
)
mask_sim = (
    (reco_sim["energy"][:, idx] > 0)
    & (reco_sim["pid"][:, idx] > 0)
    & (reco_sim["momentum"][:, idx, 1] > -1)
    & (reco_sim["energy"][:, idx] < 40)
)
print("tot events gen:", mask_gen.sum())
print("tot events sim:", mask_sim.sum())


def error_plots(gen, sim):
    col = "purple"
    reco_gen_err, reco_sim_err = np.sqrt(gen[0]), np.sqrt(sim[0])
    plt.stairs(
        gen[0] + reco_gen_err,
        edges=gen[1],
        baseline=gen[0] - reco_gen_err,
        color=col,
        alpha=0.3,
        fill=True,
    )
    plt.stairs(
        sim[0] + reco_sim_err,
        edges=sim[1],
        baseline=sim[0] - reco_sim_err,
        color="dimgray",
        hatch="///",
    )


col = "purple"
plt.figure(1, figsize=(23, 15))

plt.subplot(a, b, 1)
en_gen = reco_gen["energy"][:, idx][mask_gen]
en_sim = reco_sim["energy"][:, idx][mask_sim]
gen = plt.hist(en_gen, bins=bins_, histtype="step", label="CaloHadronic", color=col, linewidth=3)
sim = plt.hist(en_sim, bins=bins_, alpha=0.5, label="Geant4", color="gray")
error_plots(gen, sim)
plt.xlabel("Energy [GeV]")
plt.yscale("log")

ax = plt.subplot(a, b, 2)
props = dict(boxstyle="round", edgecolor=col, facecolor="white", alpha=0.5)
ax.text(
    0.1, 0.95,
    " leading PFO $E < 40$ GeV",
    transform=ax.transAxes,
    verticalalignment="top",
    fontsize=35,
    fontfamily="serif",
    bbox=props,
)
plt.axis("off")

plt.subplot(a, b, 3)
m_gen_0 = reco_gen["momentum"][:, idx, 0][mask_gen]
m_sim_0 = reco_sim["momentum"][:, idx, 0][mask_sim]
gen = plt.hist(m_gen_0, bins=bins_, histtype="step", label="gen", color=col, linewidth=3)
sim = plt.hist(m_sim_0, bins=bins_, alpha=0.5, label="sim", color="gray")
error_plots(gen, sim)
plt.xlabel("$p_{x}$ [GeV]")
plt.yscale("log")

plt.subplot(a, b, 4)
m_gen_1 = reco_gen["momentum"][:, idx, 1][mask_gen]
m_sim_1 = reco_sim["momentum"][:, idx, 1][mask_sim]
gen = plt.hist(m_gen_1, bins=bins_, histtype="step", color=col, linewidth=3, label="CaloHadronic")
sim = plt.hist(m_sim_1, bins=bins_, alpha=0.5, color="gray", label="Geant4")
error_plots(gen, sim)
plt.xlabel("$p_{y}$ [GeV]")
plt.yscale("log")
plt.legend(fontsize=30, loc="upper left")

plt.subplot(a, b, 5)
m_gen_2 = reco_gen["momentum"][:, idx, 2][mask_gen]
m_sim_2 = reco_sim["momentum"][:, idx, 2][mask_sim]
gen = plt.hist(m_gen_2, bins=bins_, histtype="step", color=col, linewidth=3)
sim = plt.hist(m_sim_2, bins=bins_, alpha=0.5, color="gray")
error_plots(gen, sim)
plt.xlabel("$p_{z}$ [GeV]")
plt.yscale("log")

plt.subplot(a, b, 6)
pid_gen = reco_gen["pid"][:, idx][mask_gen]
pid_sim = reco_sim["pid"][:, idx][mask_sim]
mapped_values_gen = np.array([pid_to_index.get(pid, -1) for pid in pid_gen])
mapped_values_sim = np.array([pid_to_index.get(pid, -1) for pid in pid_sim])
gen = plt.hist(
    mapped_values_sim,
    bins=np.arange(-0.5, len(pid_to_index) + 0.5, 1),
    range=[0, mapped_values_sim.max() + 1],
    alpha=0.5,
    color="gray",
)
sim = plt.hist(
    mapped_values_gen,
    bins=np.arange(-0.5, len(pid_to_index) + 0.5, 1),
    range=[0, mapped_values_gen.max() + 1],
    histtype="step",
    color=col,
    linewidth=3,
)
error_plots(gen, sim)
plt.yscale("log")
tick_positions = list(dict_pid_labels.values())
tick_labels = list(dict_pid_labels.keys())
plt.xticks(tick_positions, tick_labels, rotation=45)

plt.tight_layout()
out = "/data/dust/user/mmozzani/pion-clouds/figs/reconstruction/reco_leading_energy_lt40.pdf"
print(out)
plt.savefig(out)
plt.close()
