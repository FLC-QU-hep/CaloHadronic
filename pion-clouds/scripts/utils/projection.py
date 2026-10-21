import matplotlib as mpl
import numpy as np
from scipy.stats import binned_statistic_dd
from tqdm import tqdm

mpl.rcParams["xtick.labelsize"] = 25
mpl.rcParams["ytick.labelsize"] = 25
# mpl.rcParams['font.size'] = 28
mpl.rcParams["font.size"] = 35
mpl.rcParams["legend.frameon"] = False
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
# mpl.rcParams['axes.aspect'] = 'equal'
# mpl.rcParams['figure.autolayout'] = True # for the hit e distr.
# mpl.rcParams['font.weight'] = 1  # Or you can use a numerical value, e.g., 700

X = np.load("/data/dust/user/mmozzani/pion-clouds/scripts/create_data/muon-map/X.npy")
Z = np.load("/data/dust/user/mmozzani/pion-clouds/scripts/create_data/muon-map/Z.npy")
Y = np.load("/data/dust/user/mmozzani/pion-clouds/scripts/create_data/muon-map/Y.npy")
E = np.load("/data/dust/user/mmozzani/pion-clouds/scripts/create_data/muon-map/E.npy")
layer_bottom_pos = np.load(
    "/data/dust/user/mmozzani/pion-clouds/scripts/create_data/layer_bottom_pos_ecal+hcal.npy"
)
layer_top_pos = np.load(
    "/data/dust/user/mmozzani/pion-clouds/scripts/create_data/layer_top_pos_ecal+hcal.npy"
)

Ymin = 0
Xmin, Xmax = -450, 450
Zmin, Zmax = -450, 450
half_cell_size_ecal = 5.0883331298828125 / 2
half_cell_size_hcal = 5.0883331298828125 / 2 * 12
cell_thickness_ecal = layer_top_pos[0] - layer_bottom_pos[0]  # 0.52502441406250
cell_thickness_hcal = layer_top_pos[-1] - layer_bottom_pos[-1]  # 3

inbox_idx = np.where((Y > Ymin) & (X < Xmax) & (X > Xmin) & (Z < Zmax) & (Z > Zmin))[0]
X = X[inbox_idx]
Z = Z[inbox_idx]
Y = Y[inbox_idx]
E = E[inbox_idx]


def create_map(X, Y, Z, dm=3):
    """
    X, Y, Z: np.array
        ILD coordinates of sencors hited with muons
    dm: int (1, 2, 3, 4, 5) dimension split multiplicity
    """
    half_cell_size = 5.0883331298828125 / 2
    offset = half_cell_size * 2 / (dm)
    layers = []
    for layer in tqdm(range(len(layer_bottom_pos))):  # loop over layers
        xedges = np.array([])
        zedges = np.array([])
        if layer < 30:
            half_cell_size = half_cell_size_ecal
            cell_thickness = cell_thickness_ecal
        else:
            half_cell_size = half_cell_size_hcal
            cell_thickness = cell_thickness_hcal

        idx = np.where(
            (Y <= (layer_bottom_pos[layer] + cell_thickness * 1.5))
            & (Y >= layer_bottom_pos[layer] - cell_thickness / 2)
        )
        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])

        xedges = np.append(xedges, unique_X[0] - half_cell_size)
        xedges = np.append(xedges, unique_X[0] + half_cell_size)

        for i in range(len(unique_X) - 1):  # loop over X coordinate cell centers
            if abs(unique_X[i] - unique_X[i + 1]) > half_cell_size * 1.9:
                xedges = np.append(xedges, unique_X[i + 1] - half_cell_size)
                xedges = np.append(xedges, unique_X[i + 1] + half_cell_size)

                for of_m in range(dm):
                    xedges = np.append(
                        xedges, unique_X[i + 1] - half_cell_size + offset * of_m
                    )  # for higher granularity

        for z in unique_Z:  # loop over Z coordinate cell centers
            zedges = np.append(zedges, z - half_cell_size)
            zedges = np.append(zedges, z + half_cell_size)

            for of_m in range(dm):
                zedges = np.append(
                    zedges, z - half_cell_size + offset * of_m
                )  # for higher granularity

        zedges = np.unique(zedges)
        xedges = np.unique(xedges)

        xedges = [
            xedges[i]
            for i in range(len(xedges) - 1)
            if abs(xedges[i] - xedges[i + 1]) > 1e-3
        ] + [xedges[-1]]
        zedges = [
            zedges[i]
            for i in range(len(zedges) - 1)
            if abs(zedges[i] - zedges[i + 1]) > 1e-3
        ] + [zedges[-1]]

        H, xedges, zedges = np.histogram2d(X[idx], Z[idx], bins=(xedges, zedges))
        layers.append({"xedges": xedges, "zedges": zedges, "grid": H})

    return layers, offset


def projection(showers, with_time=False):
    events = []
    energy = []
    for i in tqdm(
        range(showers.shape[0]), mininterval=5, desc="Projecting..."
    ):  # real_showers.shape[0]):
        one_shw = np.moveaxis(showers[i], 0, 1)
        H, my_edges = np.histogramdd(
            one_shw[:, :3], bins=[30, 30, 78], weights=one_shw[:, 3]
        )
        # edges.append(my_edges)
        # energy.append(one_shw[:,3])
        events.append(H)
        del H, my_edges, one_shw
    return events, energy


def merge_ecal_and_hcal(H_ecal, H_hcal):
    zeros_pad = int((H_ecal.shape[0] - H_hcal.shape[0]) / 2)
    H_hcal = np.append(
        H_hcal, np.zeros((zeros_pad, H_hcal.shape[1], H_hcal.shape[2])), axis=0
    )
    H_hcal = np.append(
        np.zeros((zeros_pad, H_hcal.shape[1], H_hcal.shape[2])), H_hcal, axis=0
    )
    H_hcal = np.append(
        H_hcal, np.zeros((H_hcal.shape[0], H_hcal.shape[1], zeros_pad)), axis=2
    )
    H_hcal = np.append(
        np.zeros((H_hcal.shape[0], H_hcal.shape[1], zeros_pad)), H_hcal, axis=2
    )
    H = np.concatenate([H_ecal, H_hcal], axis=1)
    return H


# @profile
def projection_ecal_hcal(
    showers,
    x_bins,
    x_bins_hcal,
    y_bins,
    y_bins_ecal,
    y_bins_hcal,
    z_bins,
    z_bins_hcal,
    with_time=False,
):
    s = int(x_bins.shape[0] - 1)
    ys, _, _ = (
        int(y_bins.shape[0] - 1),
        int(y_bins_ecal.shape[0] - 1),
        int(y_bins_hcal.shape[0] - 1),
    )
    events = np.zeros(((showers.shape[0], s, ys, s)))
    events_time = np.zeros(((showers.shape[0], s, ys, s)))
    # events_ecal = np.zeros(((showers.shape[0], s, ys_ecal, s)))
    # events_hcal = np.zeros(((showers.shape[0], s, ys_hcal, s)))
    for i in tqdm(
        range(showers.shape[0]), mininterval=10, desc="Projecting..."
    ):  # real_showers.shape[0]):
        one_shw = np.moveaxis(showers[i], 0, 1)
        ecal_shw = one_shw[one_shw[:, 1] < 30, :]
        hcal_shw = one_shw[one_shw[:, 1] >= 30, :]

        # ECAL projection ----------------------------------
        random_shift_ecal = np.random.uniform(0, 5)
        H_ecal, _ = np.histogramdd(
            ecal_shw[:, :3],
            bins=[x_bins + random_shift_ecal, y_bins_ecal, z_bins + random_shift_ecal],
            weights=ecal_shw[:, 3],
        )

        # HCAL projection ----------------------------------
        random_shift_hcal = np.random.uniform(0, 30)
        H_hcal, _ = np.histogramdd(
            hcal_shw[:, :3],
            bins=[
                x_bins_hcal + random_shift_hcal,
                y_bins_hcal,
                z_bins_hcal + random_shift_hcal,
            ],
            weights=hcal_shw[:, 3],
        )

        # merge ECAL and HCAL ------------------------------
        H = merge_ecal_and_hcal(H_ecal, H_hcal)
        events[i] = H
        del H, H_ecal, H_hcal

        if with_time:
            if ecal_shw[:, :3].shape[0] == 0:
                H_time_ecal = np.full(
                    (x_bins, y_bins, z_bins), np.nan
                )  # or +inf or 0 depending on convention
            else:
                H_time_ecal, _, _ = binned_statistic_dd(
                    ecal_shw[:, :3],
                    ecal_shw[:, 4],
                    statistic="min",
                    bins=[
                        x_bins + random_shift_ecal,
                        y_bins_ecal,
                        z_bins + random_shift_ecal,
                    ],
                )

            if hcal_shw[:, :3].shape[0] == 0:
                H_time_hcal = np.full(
                    (len(x_bins_hcal) + 1, len(y_bins_hcal) - 1, len(z_bins_hcal) + 1),
                    np.nan,
                )  # or +inf or 0 depending on convention
            else:
                H_time_hcal, _, _ = binned_statistic_dd(
                    hcal_shw[:, :3],
                    hcal_shw[:, 4],
                    statistic="min",
                    bins=[
                        x_bins_hcal + random_shift_hcal,
                        y_bins_hcal,
                        z_bins_hcal + random_shift_hcal,
                    ],
                )

            H_time = merge_ecal_and_hcal(H_time_ecal, H_time_hcal)
            events_time[i] = H_time
            del H_time, H_time_ecal, H_time_hcal

        # events_ecal[i] = H_ecal
        # events_hcal[i] = H_hcal
        del one_shw, ecal_shw, hcal_shw
    del showers
    return events, events_time


def remove_low_energy_SoftCut(event):
    events = event.copy()
    if events[:, 3].max() < 1:
        divide = 1000
    else:
        divide = 1

    for i in range(len(events)):
        events[i] = SoftCut_new(
            np.array(events[i]), min_en=pow(10, -3.7) / divide, max_en=0.01 / divide
        )
    # removing a small peak that causes prolems-------------------------
    pos_row, pos_col = np.where(
        (events[:, 3, :] < 3.73e-3) & (events[:, 3, :] > 3.65e-3)
    )
    for coord in range(4):
        events[pos_row[3:], coord, pos_col[3:]] = 0
    # print('canceled events at the peak:', pos_row[3:].shape)
    # -------------------------------------------------------------------
    for i in range(len(events)):
        indx = np.argsort(events[i, -1])  # then I need to sort again!!
        for coordinate in range(4):
            events[i, coordinate] = events[i, coordinate][
                indx
            ]  # now all the zeros are at the beginning
    return events


def SoftCut_new(event, min_en, max_en):
    my_pos = np.where(event[3] <= min_en)[0]
    for coord in range(4):
        event[coord, my_pos] = 0
    to_softcut = np.where((event[3] > min_en) & (event[3] < max_en))[0]
    if len(to_softcut) == 0:
        return event
    else:
        energy_toSoftCut = event[3][to_softcut]
        mapping = dict(zip(energy_toSoftCut, to_softcut))
        r = -np.exp(-pow(energy_toSoftCut, 2))
        prob = (r - np.min(r)) / (np.max(r) - np.min(r))
        for j in range(len(prob)):
            if prob[j] < np.random.rand():
                for coord in range(4):
                    event[coord][mapping[energy_toSoftCut[j]]] = (
                        0  # cutting the E,X,Y,Z
                    )
        return event


def cluster_hits_allCAL(
    event, cell_size=6
):  # this is 2.5 cm x 6 = 15 cm cell size; if cell_size = 1 the default space is 2.5 cm
    Ymax = 77
    y_hcal_bins = np.arange(30, Ymax + 1, 1)
    # new just for the hcal part
    x_bins = np.arange(Xmin, Xmax + cell_size, cell_size)
    z_bins = np.arange(Zmin, Zmax + cell_size, cell_size)
    pad_len = event.shape[-1]
    max_len = (event[:, -1] > 0).sum(axis=1).max()
    event = event[:, :, -max_len:]
    new_event = np.zeros((event.shape[0], 4, pad_len))

    for j in range(event.shape[0]):
        ecal_shw = event[j][:, event[j, 1] < 30]
        ecal_shw = ecal_shw[:, ecal_shw[-1] > 0]
        to_cluster = event[j][:, event[j, 1] >= 30]
        H, edges = np.histogramdd(
            to_cluster[:3].T, bins=[x_bins, y_hcal_bins, z_bins], weights=to_cluster[-1]
        )
        mask = H != 0
        nr_edge_x, nr_edge_y, nr_edge_z = np.where(
            mask
        )  # edges idx where mask in not zero
        e = H[mask]
        # x and z coordinates of the point at the center of the cell
        x = (edges[0][nr_edge_x] + edges[0][nr_edge_x + 1]) / 2
        z = (edges[2][nr_edge_z] + edges[2][nr_edge_z + 1]) / 2
        y = edges[1][nr_edge_y]
        hcal_shw = np.stack((x, y, z, e))
        tot_shw = np.concatenate((ecal_shw, hcal_shw), axis=1)
        new_event[j] = np.concatenate(
            (tot_shw, np.zeros((4, pad_len - tot_shw.shape[1]))), axis=1
        )

    max_len = (new_event[:, -1] > 0).sum(axis=1).max()
    new_event = new_event[:, :, :max_len]
    for g in [0, 2]:
        cut = new_event[:, 1] >= 30
        new_event[:, g][cut] = (
            new_event[:, g][cut]
            + np.random.randint(10, size=new_event[:, g][cut].shape) / 100
        )
    return new_event
