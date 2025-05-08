from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
import ot

__all__ = ["heun_integrate", "CNF", "FullyConnected", "ConcatSquash"]

class FullyConnected(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_cond_inputs: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = num_inputs + num_cond_inputs + dim_time
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_inputs))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        input = [t, x]
        if condition is not None:
            input.append(condition)
        input = torch.cat(input, dim=-1)
        return self.network(input)


class ConcatSquash(nn.Module):
    class __ConcatSquashLayer(nn.Module):
        def __init__(
            self,
            dim_input: int,
            dim_output: int,
            dim_cond: int,
            activation: type[nn.Module],
        ) -> None:
            super().__init__()
            self.cond_embed = nn.Sequential(
                nn.Linear(dim_cond, dim_input),
            )
            self.network = nn.Sequential(
                nn.Linear(dim_input, dim_output),
                activation(),
            )

        def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            cond_embed = self.cond_embed(condition)
            return self.network(x + cond_embed)

    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = dim_input
        dim_cond_time = dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(
                self.__ConcatSquashLayer(prev_dim, dim, dim_cond_time, activation)
            )
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, dim_input)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        condition = torch.cat([t, condition], dim=-1)
        for layer in self.layers:
            x = layer(x, condition)
        return self.output(x)
    
    
def heun_integrate(
    ode: Callable[[Tensor, Tensor], Tensor],
    x0: Tensor,
    t0: float,
    t1: float,
    steps: int,
    **kwargs,
) -> Tensor:
    t0_ = torch.as_tensor(t0, dtype=x0.dtype, device=x0.device)
    t1_ = torch.as_tensor(t1, dtype=x0.dtype, device=x0.device)

    x = x0
    t = t0_
    dt = (t1_ - t0_) / steps
    for _ in range(steps):
        df = ode(t, x, **kwargs)
        y_ = x + dt * df
        x = x + dt / 2 * (df + ode(t + dt, y_, **kwargs))
        t = t + dt

    return x


# partially based on https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
class CNF(nn.Module):
    def __init__(
        self,
        network: Callable[[Tensor, Tensor], Tensor],
        frequencies: int = 3,
        acceleration: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = network
        self.acceleration = acceleration
        self.register_buffer(
            name="frequencies",
            tensor=(torch.arange(1, frequencies + 1) * torch.pi).reshape(1, -1),
        )

    def forward(self, t: Tensor, x: Tensor, **kwargs) -> Tensor:
        t = self.time_trafo(t)
        t = self.frequencies * t.reshape(-1, 1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(x.shape[0], -1)

        return self.network(t, x, **kwargs)

    def encode(self, x: Tensor, num_timesteps: int = 200, **kwargs) -> Tensor:
        return heun_integrate(self, x, 0.0, 1.0, num_timesteps, **kwargs)

    def decode(self, z: Tensor, num_timesteps: int = 200, **kwargs) -> Tensor:
        return heun_integrate(self, z, 1.0, 0.0, num_timesteps, **kwargs)

    def time_trafo(self, t: Tensor) -> Tensor:
        return (1 - self.acceleration) * t + self.acceleration * t**2

    def time_derivative(self, t: Tensor) -> Tensor:
        return 2 * self.acceleration * t - self.acceleration + 1

    def loss(self, x: Tensor, noise: Tensor | None = None, **kwargs) -> Tensor:
        t = torch.rand(
            [x.shape[0]] + [1] * (x.dim() - 1), device=x.device, dtype=x.dtype
        )
        t_ = self.time_trafo(t)
        if noise is not None:
            z = noise
        else:
            z = torch.randn_like(x)
        y = (1 - t_) * x + (1e-4 + (1 - 1e-4) * t_) * z
        u = (1 - 1e-4) * z - x
        u = self.time_derivative(t) * u

        return (self(t.reshape(-1, 1), y, **kwargs) - u).square()

    def ot_loss(self, x: Tensor, noise: Tensor | None = None, **kwargs) -> Tensor:
        t = torch.rand(
            [x.shape[0]] + [1] * (x.dim() - 1), device=x.device, dtype=x.dtype
        )
        t_ = self.time_trafo(t)
        
        N = x.shape[0]
        if noise is not None:
            z = noise
        else:
            z = torch.randn_like(x)
        M = torch.sqrt(torch.sum((x[:, None, :] - z[None, :, :]) ** 2, dim=-1))
        wa = torch.ones(N) / N
        wb = torch.ones(N) / N
        T = ot.emd(wa, wb, M, numItermax=1_000_000).to(x.device)
        z = N * (T @ z)
        
        x_t = (1 - t) * x + t * z
        v_target = (z - x)  
        return ((self(t_, x_t, **kwargs)  - v_target) ** 2).mean()

    def sample(
        self, shape: tuple[int, ...], num_timesteps: int = 200, **kwargs
    ) -> Tensor:
        z = torch.randn(
            *shape, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self.decode(z, num_timesteps, **kwargs)

    def __repr__(self) -> str:
        network = self.network.__repr__().replace("\n", "\n  ")
        return f"{self.__class__.__name__}(\n  (network): {network}\n  frequencies/pi={(self.frequencies[0] / torch.pi).tolist()},\n  acceleration={self.acceleration}\n)"