import torch
import torch.nn as nn

from pyro.nn import ConditionalDenseNN, DenseNN
import pyro.distributions as dist
import pyro.distributions.transforms as T
from .custom_pyro import ConditionalAffineCouplingTanH


# a context manager to fix a seed
class seed_torch:
    def __init__(self, seed=42):
        self.seed = seed

    def __enter__(self):
        self.previous_seed = torch.seed()
        torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        torch.manual_seed(self.previous_seed)


def get_gauss_basis(num_inputs, device, **kwargs):
    """
    Create a multivariate Gaussian distribution.

    Parameters
    ----------
    num_inputs : int
        Dimension of the input and output space.
    device : torch.device
        The device on which the model will be run, either 'cpu' or 'cuda'.

    Returns
    -------
    base_dist : dist.Normal
        The distribution created by the transformations.

    """
    base_dist = dist.Normal(
        torch.zeros(num_inputs).to(device), torch.ones(num_inputs).to(device)
    )
    return base_dist


class SafeExpTransform(T.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.nonnegative
    bijective = True
    sign = +1

    def __init__(self, eps=1e-6, cache_size=1):
        super().__init__(cache_size=cache_size)
        self.eps = eps

    def _call(self, x):
        return torch.exp(x) - self.eps

    def _inverse(self, y):
        return torch.log(y + self.eps)

    def log_abs_det_jacobian(self, x, y):
        return x


class HybridTanH_factory:
    def __init__(self, num_inputs, num_cond_inputs, device):
        self.num_inputs = num_inputs
        self.num_cond_inputs = num_cond_inputs
        self.device = device

    def add_permutation(self, **kwargs):
        perm = torch.randperm(self.num_inputs, dtype=torch.long).to(self.device)
        ff = T.Permute(perm)
        self.transforms.append(ff)

    def add_affine_coupling(self, **kwargs):
        split_dim = self.num_inputs // 2
        param_dims = [self.num_inputs - split_dim, self.num_inputs - split_dim]
        hypernet = ConditionalDenseNN(
            split_dim,
            self.num_cond_inputs,
            [self.num_inputs * 10, self.num_inputs * 10],
            param_dims,
        )
        ctf = ConditionalAffineCouplingTanH(split_dim, hypernet).to(self.device)
        self.trainables.append(ctf)
        self.transforms.append(ctf)

    def add_spline_coupling(self, count_bins, **kwargs):
        param_dims = [
            self.num_inputs * count_bins,
            self.num_inputs * count_bins,
            self.num_inputs * (count_bins - 1),
            self.num_inputs * count_bins,
        ]
        hypernet = DenseNN(
            self.num_cond_inputs, [self.num_inputs * 4, self.num_inputs * 4], param_dims
        )
        ctf = T.ConditionalSpline(hypernet, self.num_inputs, count_bins).to(self.device)
        self.trainables.append(ctf)
        self.transforms.append(ctf)

    def add_partial_log(self, **kwargs):
        # Log transform the last 60 numbers
        log_len = 60
        prefix_len = self.num_inputs - log_len
        exp_transform = SafeExpTransform()
        transform = T.CatTransform(
            [T.identity_transform, exp_transform], dim=-1, lengths=[prefix_len, log_len]
        )
        self.transforms.append(transform)

    def add_repeat(self, transform_pattern, transform_args):
        for trans in transform_pattern:
            if isinstance(trans, str):
                trans = getattr(self, "add_" + trans)
            trans(**transform_args)

    def create(
        self,
        num_repeats,
        transform_pattern,
        base_dist_gen=get_gauss_basis,
        **transform_args
    ):
        # TODO, could the params of the base dist also be added to the model?
        with seed_torch(42):
            base_dist = base_dist_gen(self.num_inputs, self.device, **transform_args)
            self.transforms = []
            self.trainables = []
            for _ in range(num_repeats):
                self.add_repeat(transform_pattern, transform_args)
            modules = nn.ModuleList(self.trainables)
            flow_dist = dist.ConditionalTransformedDistribution(
                base_dist, self.transforms
            )

        return modules, flow_dist, self.transforms


def compile_HybridTanH_model(num_blocks, num_inputs, num_cond_inputs, device):
    """
    Simplified version of the default

    Create a showerflow model to train, and also the distribution that it defines.
    As this is a normalising flow model, the input and output space have the same dimensions.

    It isn't hard coded, but usually there are 65 dimensions.
    In order of appearance, the dimensions encode;
    - the number of points total
    - total visible energy
    - center of gravity (energy) in x
    - center of gravity (energy) in y
    - center of gravity (energy) in z
    - number of clusters on layer 1
    ...
    - number of clusters on layer 30
    - total energy on layer 1
    ...
    - total energy on layer 30


    Parameters
    ----------
    num_blocks : int
        Number of transformaitons the model applies to the distribution.
    num_inputs : int
        Dimension of the input and output space.
    num_cond_inputs : int
        Dimension of the conditioning input. Same information is given to each transformation.
    device : torch.device
        The device on which the model will be run, either 'cpu' or 'cuda'.

    Returns
    -------
    model : nn.ModuleList
        The trainable model that will define the transformaitons.
    flow_dist : dist.ConditionalTransformedDistribution
        The distribution created by the transformations.
    transforms : list
        The list of transformations that the model applies.

    """
    factory = HybridTanH_factory(num_inputs, num_cond_inputs, device)

    transform_pattern = [
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "permutation",
        "spline_coupling",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
    ]

    model, flow_dist, transforms = factory.create(
        num_blocks, transform_pattern, count_bins=8
    )
    return model, flow_dist, transforms


def compile_HybridTanH_alt1(num_blocks, num_inputs, num_cond_inputs, device):
    factory = HybridTanH_factory(num_inputs, num_cond_inputs, device)

    transform_pattern = [
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "spline_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
    ]

    model, flow_dist, transforms = factory.create(
        num_blocks, transform_pattern, count_bins=8
    )
    return model, flow_dist, transforms


def compile_HybridTanH_log1(num_blocks, num_inputs, num_cond_inputs, device):
    factory = HybridTanH_factory(num_inputs, num_cond_inputs, device)

    transform_pattern = [
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "spline_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
    ]
    transform_pattern = transform_pattern * num_blocks
    transform_pattern += ["partial_log"]

    model, flow_dist, transforms = factory.create(1, transform_pattern, count_bins=8)
    return model, flow_dist, transforms


def compile_HybridTanH_alt2(num_blocks, num_inputs, num_cond_inputs, device):
    factory = HybridTanH_factory(num_inputs, num_cond_inputs, device)

    transform_pattern = [
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
        "affine_coupling",
        "permutation",
    ]

    model, flow_dist, transforms = factory.create(
        num_blocks, transform_pattern, count_bins=8
    )
    return model, flow_dist, transforms


versions_dict = {
    "original": compile_HybridTanH_model,
    "alt1": compile_HybridTanH_alt1,
    "log1": compile_HybridTanH_log1,
    "alt2": compile_HybridTanH_alt2,
}
