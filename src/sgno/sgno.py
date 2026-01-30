from __future__ import annotations

import math
import re
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn


def _inv_softplus(x: float, eps: float = 1e-6) -> float:
    x = float(max(x, eps))
    return math.log(math.expm1(x))


def _phi1(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    small = torch.abs(z) < eps
    series = 1.0 + z * 0.5 + (z * z) / 6.0
    safe = (torch.exp(z) - 1.0) / z
    return torch.where(small, series, safe)


def _smooth_filter_nd(shape, strength: float, order: int, device, dtype=torch.float32):
    if strength <= 0:
        return torch.ones(*shape, device=device, dtype=dtype)
    grids = []
    for m in shape:
        if m <= 1:
            grids.append(torch.zeros(m, device=device, dtype=dtype))
        else:
            grids.append(torch.linspace(0.0, 1.0, steps=m, device=device, dtype=dtype))
    mesh = torch.meshgrid(*grids, indexing='ij')
    r2 = torch.zeros_like(mesh[0])
    for g in mesh:
        r2 = r2 + g * g
    r = torch.sqrt(r2 + 1e-12)
    return torch.exp(-float(strength) * (r ** int(order)))


def _two_thirds_mask_1d(m1: int, device, dtype=torch.float32):
    if m1 <= 1:
        return torch.ones(m1, device=device, dtype=dtype)
    cutoff = max(1, int(math.floor((2.0 / 3.0) * (m1 - 1))))
    k = torch.arange(m1, device=device)
    return (k <= cutoff).to(dtype)


def _two_thirds_mask_2d(m1: int, m2: int, device, dtype=torch.float32):
    return _two_thirds_mask_1d(m1, device, dtype)[:, None] * _two_thirds_mask_1d(m2, device, dtype)[None, :]


def _two_thirds_mask_3d(m1: int, m2: int, m3: int, device, dtype=torch.float32):
    return (
        _two_thirds_mask_1d(m1, device, dtype)[:, None, None]
        * _two_thirds_mask_1d(m2, device, dtype)[None, :, None]
        * _two_thirds_mask_1d(m3, device, dtype)[None, None, :]
    )


class PointwiseMLP1d(nn.Module):
    def __init__(self, width: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or (2 * width)
        self.net = nn.Sequential(
            nn.Conv1d(width, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, width, 1),
        )

    def forward(self, x):
        return self.net(x)


class PointwiseMLP2d(nn.Module):
    def __init__(self, width: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or (2 * width)
        self.net = nn.Sequential(
            nn.Conv2d(width, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, width, 1),
        )

    def forward(self, x):
        return self.net(x)


class PointwiseMLP3d(nn.Module):
    def __init__(self, width: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or (2 * width)
        self.net = nn.Sequential(
            nn.Conv3d(width, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, width, 1),
        )

    def forward(self, x):
        return self.net(x)


class SpectralETD1d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes1: int,
        dt: float = 1.0,
        use_beta: bool = False,
        filter_type: str = 'smooth',
        filter_strength: float = 2.0,
        filter_order: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.modes1 = modes1
        self.dt = float(dt)
        self.use_beta = bool(use_beta)
        self.filter_type = str(filter_type)
        self.filter_strength = float(filter_strength)
        self.filter_order = int(filter_order)

        self.log_decay = nn.Parameter(torch.randn(channels, modes1) * 0.1)
        if self.use_beta:
            self.beta = nn.Parameter(torch.randn(channels, modes1) * 0.1)
        else:
            self.register_parameter('beta', None)

        scale = 1.0 / (channels * channels)
        self.mix = nn.Parameter(scale * torch.rand(channels, channels, modes1, dtype=torch.cfloat))

    def _lambda(self, device):
        alpha = -F.softplus(self.log_decay)
        if self.use_beta and self.beta is not None:
            beta = self.beta
        else:
            beta = torch.zeros_like(alpha)
        return torch.complex(alpha, beta).to(device=device)

    def _filter(self, device):
        if self.filter_type == 'none':
            return None
        if self.filter_type in ('2/3', 'two_thirds', 'two-thirds'):
            return _two_thirds_mask_1d(self.modes1, device=device)
        return _smooth_filter_nd((self.modes1,), self.filter_strength, self.filter_order, device=device)

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, C, X = v.shape
        assert C == self.channels

        v_hat = torch.fft.rfft(v.float())
        g_hat = torch.fft.rfft(g.float())
        out_hat = torch.zeros_like(v_hat)

        m1 = min(self.modes1, v_hat.size(-1))

        lam = self._lambda(v.device)[:, :m1]
        z = (self.dt * lam).unsqueeze(0)
        expz = torch.exp(z)
        phi = _phi1(z)

        vh = v_hat[:, :, :m1]
        gh = g_hat[:, :, :m1]

        mix = self.mix[:, :, :m1].to(device=v.device)
        gmix = torch.einsum('bim,iom->bom', gh, mix)

        lin = expz * vh
        forcing = (self.dt * phi) * gmix

        filt = self._filter(v.device)
        if filt is not None:
            forcing = forcing * filt[:m1].view(1, 1, m1)

        out_hat[:, :, :m1] = lin + forcing
        return torch.fft.irfft(out_hat, n=X)


class SpectralETD2d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        dt: float = 1.0,
        use_beta: bool = False,
        filter_type: str = 'smooth',
        filter_strength: float = 2.0,
        filter_order: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.dt = float(dt)
        self.use_beta = bool(use_beta)
        self.filter_type = str(filter_type)
        self.filter_strength = float(filter_strength)
        self.filter_order = int(filter_order)

        self.log_decay_pos = nn.Parameter(torch.randn(channels, modes1, modes2) * 0.1)
        self.log_decay_neg = nn.Parameter(torch.randn(channels, modes1, modes2) * 0.1)

        if self.use_beta:
            self.beta_pos = nn.Parameter(torch.randn(channels, modes1, modes2) * 0.1)
            self.beta_neg = nn.Parameter(torch.randn(channels, modes1, modes2) * 0.1)
        else:
            self.register_parameter('beta_pos', None)
            self.register_parameter('beta_neg', None)

        scale = 1.0 / math.sqrt(channels)
        self.mix_pos = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2, dtype=torch.cfloat))
        self.mix_neg = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2, dtype=torch.cfloat))

    def _lambda(self, device):
        a_pos = -F.softplus(self.log_decay_pos)
        a_neg = -F.softplus(self.log_decay_neg)
        if self.use_beta and self.beta_pos is not None and self.beta_neg is not None:
            b_pos = self.beta_pos
            b_neg = self.beta_neg
        else:
            b_pos = torch.zeros_like(a_pos)
            b_neg = torch.zeros_like(a_neg)
        lam_pos = torch.complex(a_pos, b_pos).to(device=device)
        lam_neg = torch.complex(a_neg, b_neg).to(device=device)
        return lam_pos, lam_neg

    def _filter(self, device):
        if self.filter_type == 'none':
            return None
        if self.filter_type in ('2/3', 'two_thirds', 'two-thirds'):
            return _two_thirds_mask_2d(self.modes1, self.modes2, device=device)
        return _smooth_filter_nd((self.modes1, self.modes2), self.filter_strength, self.filter_order, device=device)

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, C, X, Y = v.shape
        assert C == self.channels

        v_hat = torch.fft.rfft2(v.float())
        g_hat = torch.fft.rfft2(g.float())
        out_hat = torch.zeros_like(v_hat)

        m1 = min(self.modes1, v_hat.size(-2))
        m2 = min(self.modes2, v_hat.size(-1))

        lam_pos, lam_neg = self._lambda(v.device)
        lam_pos = lam_pos[:, :m1, :m2]
        lam_neg = lam_neg[:, :m1, :m2]

        z_pos = (self.dt * lam_pos).unsqueeze(0)
        z_neg = (self.dt * lam_neg).unsqueeze(0)
        exp_pos = torch.exp(z_pos)
        exp_neg = torch.exp(z_neg)
        phi_pos = _phi1(z_pos)
        phi_neg = _phi1(z_neg)

        filt = self._filter(v.device)
        if filt is not None:
            filt = filt[:m1, :m2].view(1, 1, m1, m2)

        vp = v_hat[:, :, :m1, :m2]
        gp = g_hat[:, :, :m1, :m2]
        mixp = self.mix_pos[:, :, :m1, :m2].to(device=v.device)
        gmp = torch.einsum('bixy,ioxy->boxy', gp, mixp)
        lin_p = exp_pos * vp
        forcing_p = (self.dt * phi_pos) * gmp
        if filt is not None:
            forcing_p = forcing_p * filt
        out_hat[:, :, :m1, :m2] = lin_p + forcing_p

        vn = v_hat[:, :, -m1:, :m2]
        gn = g_hat[:, :, -m1:, :m2]
        mixn = self.mix_neg[:, :, :m1, :m2].to(device=v.device)
        gmn = torch.einsum('bixy,ioxy->boxy', gn, mixn)
        lin_n = exp_neg * vn
        forcing_n = (self.dt * phi_neg) * gmn
        if filt is not None:
            forcing_n = forcing_n * filt
        out_hat[:, :, -m1:, :m2] = lin_n + forcing_n

        return torch.fft.irfft2(out_hat, s=(X, Y))


class SpectralETD3d(nn.Module):
    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        dt: float = 1.0,
        use_beta: bool = False,
        filter_type: str = 'smooth',
        filter_strength: float = 2.0,
        filter_order: int = 8,
    ):
        super().__init__()
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.dt = float(dt)
        self.use_beta = bool(use_beta)
        self.filter_type = str(filter_type)
        self.filter_strength = float(filter_strength)
        self.filter_order = int(filter_order)

        self.log_decay1 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)
        self.log_decay2 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)
        self.log_decay3 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)
        self.log_decay4 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)

        if self.use_beta:
            self.beta1 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)
            self.beta2 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)
            self.beta3 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)
            self.beta4 = nn.Parameter(torch.randn(channels, modes1, modes2, modes3) * 0.1)
        else:
            self.register_parameter('beta1', None)
            self.register_parameter('beta2', None)
            self.register_parameter('beta3', None)
            self.register_parameter('beta4', None)

        scale = 1.0 / (channels * channels)
        self.mix1 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.mix2 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.mix3 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.mix4 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def _lam_block(self, log_decay, beta, device):
        alpha = -F.softplus(log_decay)
        if self.use_beta and beta is not None:
            b = beta
        else:
            b = torch.zeros_like(alpha)
        return torch.complex(alpha, b).to(device=device)

    def _filter(self, device):
        if self.filter_type == 'none':
            return None
        if self.filter_type in ('2/3', 'two_thirds', 'two-thirds'):
            return _two_thirds_mask_3d(self.modes1, self.modes2, self.modes3, device=device)
        return _smooth_filter_nd(
            (self.modes1, self.modes2, self.modes3),
            self.filter_strength,
            self.filter_order,
            device=device,
        )

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = v.shape
        assert C == self.channels

        v_hat = torch.fft.rfftn(v.float(), dim=[-3, -2, -1])
        g_hat = torch.fft.rfftn(g.float(), dim=[-3, -2, -1])
        out_hat = torch.zeros_like(v_hat)

        m1 = min(self.modes1, v_hat.size(-3))
        m2 = min(self.modes2, v_hat.size(-2))
        m3 = min(self.modes3, v_hat.size(-1))

        filt = self._filter(v.device)
        if filt is not None:
            filt = filt[:m1, :m2, :m3].view(1, 1, m1, m2, m3)

        b1 = None if self.beta1 is None else self.beta1[:, :m1, :m2, :m3]
        b2 = None if self.beta2 is None else self.beta2[:, :m1, :m2, :m3]
        b3 = None if self.beta3 is None else self.beta3[:, :m1, :m2, :m3]
        b4 = None if self.beta4 is None else self.beta4[:, :m1, :m2, :m3]

        lam1 = self._lam_block(self.log_decay1[:, :m1, :m2, :m3], b1, v.device)
        lam2 = self._lam_block(self.log_decay2[:, :m1, :m2, :m3], b2, v.device)
        lam3 = self._lam_block(self.log_decay3[:, :m1, :m2, :m3], b3, v.device)
        lam4 = self._lam_block(self.log_decay4[:, :m1, :m2, :m3], b4, v.device)

        def apply_block(vs, gs, lam, mix):
            zc = (self.dt * lam).unsqueeze(0)
            lin = torch.exp(zc) * vs
            mix = mix[:, :, :m1, :m2, :m3].to(device=v.device)
            gmix = torch.einsum('bixyz,ioxyz->boxyz', gs, mix)
            forcing = (self.dt * _phi1(zc)) * gmix
            if filt is not None:
                forcing = forcing * filt
            return lin + forcing

        v1 = v_hat[:, :, :m1, :m2, :m3]
        g1 = g_hat[:, :, :m1, :m2, :m3]
        out_hat[:, :, :m1, :m2, :m3] = apply_block(v1, g1, lam1, self.mix1)

        v2 = v_hat[:, :, -m1:, :m2, :m3]
        g2 = g_hat[:, :, -m1:, :m2, :m3]
        out_hat[:, :, -m1:, :m2, :m3] = apply_block(v2, g2, lam2, self.mix2)

        v3 = v_hat[:, :, :m1, -m2:, :m3]
        g3 = g_hat[:, :, :m1, -m2:, :m3]
        out_hat[:, :, :m1, -m2:, :m3] = apply_block(v3, g3, lam3, self.mix3)

        v4 = v_hat[:, :, -m1:, -m2:, :m3]
        g4 = g_hat[:, :, -m1:, -m2:, :m3]
        out_hat[:, :, -m1:, -m2:, :m3] = apply_block(v4, g4, lam4, self.mix4)

        return torch.fft.irfftn(out_hat, s=(X, Y, Z))


class SGNO1d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        modes: int = 16,
        width: int = 64,
        initial_step: int = 10,
        dt: float = 1.0,
        use_beta: bool = False,
        filter_type: str = 'smooth',
        filter_strength: float = 2.0,
        filter_order: int = 8,
        padding: int = 2,
        n_blocks: int = 4,
        alpha_w: float = 0.6,
        alpha_g: float = 1.0,
        inner_steps: int = 1,
    ):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.padding = int(padding)
        self.n_blocks = int(n_blocks)

        self.alpha_w_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_w), dtype=torch.float32))
        self.alpha_g_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_g), dtype=torch.float32))

        self.fc0 = nn.Linear(initial_step * num_channels + 1, self.width)

        self.gs = nn.ModuleList([PointwiseMLP1d(self.width) for _ in range(self.n_blocks)])
        self.etds = nn.ModuleList(
            [
                SpectralETD1d(self.width, self.modes1, dt, use_beta, filter_type, filter_strength, filter_order)
                for _ in range(self.n_blocks)
            ]
        )
        self.ws = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.n_blocks)])

        self.inner_steps = int(inner_steps)
        self._base_dts = [float(m.dt) for m in self.etds]

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding])

        alpha_w = F.softplus(self.alpha_w_raw)
        alpha_g = F.softplus(self.alpha_g_raw)

        for i in range(self.n_blocks):
            etd = self.etds[i]
            dt0 = self._base_dts[i]
            k = max(1, int(self.inner_steps))
            dt_step = dt0 / float(k)
            w_scale = 1.0 / float(k)

            for s in range(k):
                etd.dt = dt_step
                g = alpha_g * self.gs[i](x)
                upd = etd(x, g) + (alpha_w * w_scale) * self.ws[i](x)

                last = (i == self.n_blocks - 1) and (s == k - 1)
                x = upd if last else F.gelu(upd)

            etd.dt = dt0

        if self.padding > 0:
            x = x[..., :-self.padding]

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)


class SGNO2d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 20,
        initial_step: int = 10,
        dt: float = 1.0,
        use_beta: bool = False,
        filter_type: str = 'smooth',
        filter_strength: float = 2.0,
        filter_order: int = 8,
        padding: int = 2,
        n_blocks: int = 4,
        alpha_w: float = 1.0,
        alpha_g: float = 10.0,
        inner_steps: int = 1,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = int(padding)
        self.n_blocks = int(n_blocks)

        self.alpha_w_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_w), dtype=torch.float32))
        self.alpha_g_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_g), dtype=torch.float32))

        self.fc0 = nn.Linear(initial_step * num_channels + 2, self.width)

        self.gs = nn.ModuleList([PointwiseMLP2d(self.width) for _ in range(self.n_blocks)])
        self.etds = nn.ModuleList(
            [
                SpectralETD2d(
                    self.width,
                    self.modes1,
                    self.modes2,
                    dt,
                    use_beta,
                    filter_type,
                    filter_strength,
                    filter_order,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_blocks)])

        self.inner_steps = int(inner_steps)
        self._base_dts = [float(m.dt) for m in self.etds]

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        alpha_w = F.softplus(self.alpha_w_raw)
        alpha_g = F.softplus(self.alpha_g_raw)

        for i in range(self.n_blocks):
            etd = self.etds[i]
            dt0 = self._base_dts[i]
            k = max(1, int(self.inner_steps))
            dt_step = dt0 / float(k)
            w_scale = 1.0 / float(k)

            for s in range(k):
                etd.dt = dt_step
                g = alpha_g * self.gs[i](x)
                upd = etd(x, g) + (alpha_w * w_scale) * self.ws[i](x)

                last = (i == self.n_blocks - 1) and (s == k - 1)
                x = upd if last else F.gelu(upd)

            etd.dt = dt0

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)


class SGNO3d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 20,
        initial_step: int = 10,
        dt: float = 1.0,
        use_beta: bool = False,
        filter_type: str = 'smooth',
        filter_strength: float = 2.0,
        filter_order: int = 8,
        padding: int = 6,
        n_blocks: int = 4,
        alpha_w: float = 1.0,
        alpha_g: float = 10.0,
        inner_steps: int = 1,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = int(padding)
        self.n_blocks = int(n_blocks)

        self.fc0 = nn.Linear(initial_step * num_channels + 3, self.width)

        self.gs = nn.ModuleList([PointwiseMLP3d(self.width) for _ in range(self.n_blocks)])
        self.etds = nn.ModuleList(
            [
                SpectralETD3d(
                    self.width,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                    dt,
                    use_beta,
                    filter_type,
                    filter_strength,
                    filter_order,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(self.n_blocks)])

        self.inner_steps = int(inner_steps)
        self._base_dts = [float(m.dt) for m in self.etds]

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        self.alpha_w_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_w), dtype=torch.float32))
        self.alpha_g_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_g), dtype=torch.float32))

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        alpha_w = F.softplus(self.alpha_w_raw)
        alpha_g = F.softplus(self.alpha_g_raw)

        for i in range(self.n_blocks):
            etd = self.etds[i]
            dt0 = self._base_dts[i]
            k = max(1, int(self.inner_steps))
            dt_step = dt0 / float(k)
            w_scale = 1.0 / float(k)

            for s in range(k):
                etd.dt = dt_step
                g = alpha_g * self.gs[i](x)
                upd = etd(x, g) + (alpha_w * w_scale) * self.ws[i](x)

                last = (i == self.n_blocks - 1) and (s == k - 1)
                x = upd if last else F.gelu(upd)

            etd.dt = dt0

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)


def _parse_network_config(network_config: str) -> Dict[str, Any]:
    s = (network_config or '').strip()
    if not s:
        return {}
    parts = [p.strip() for p in s.split(';') if p.strip()]
    if not parts:
        return {}

    cfg: Dict[str, Any] = {}

    has_kv = any(('=' in p) for p in parts[1:])
    if has_kv:
        for p in parts[1:]:
            if '=' not in p:
                continue
            k, v = p.split('=', 1)
            k = k.strip().lower()
            v = v.strip()
            if re.fullmatch(r'-?\d+', v):
                cfg[k] = int(v)
            elif re.fullmatch(r'-?\d+(\.\d+)?([eE]-?\d+)?', v):
                cfg[k] = float(v)
            elif v.lower() in ('true', 'false'):
                cfg[k] = (v.lower() == 'true')
            else:
                cfg[k] = v
        return cfg

    nums = []
    tail = []
    for p in parts[1:]:
        if re.fullmatch(r'-?\d+', p):
            nums.append(int(p))
        elif re.fullmatch(r'-?\d+(\.\d+)?([eE]-?\d+)?', p):
            nums.append(float(p))
        else:
            tail.append(p.lower())

    if len(nums) >= 3:
        cfg['width'] = int(nums[0])
        cfg['modes'] = int(nums[1])
        cfg['n_blocks'] = int(nums[2])
    if len(nums) >= 4 and isinstance(nums[3], (int, float)):
        cfg['initial_step'] = int(nums[3])
    if len(nums) >= 5 and isinstance(nums[4], (int, float)):
        cfg['dt'] = float(nums[4])

    if tail:
        cfg['activation'] = tail[-1]

    return cfg


def _make_grid_1d(B: int, X: int, device, dtype):
    x = torch.linspace(0.0, 1.0, steps=X, device=device, dtype=dtype).view(1, X, 1)
    return x.expand(B, X, 1)


def _make_grid_2d(B: int, X: int, Y: int, device, dtype):
    gx = torch.linspace(0.0, 1.0, steps=X, device=device, dtype=dtype)
    gy = torch.linspace(0.0, 1.0, steps=Y, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(gx, gy, indexing='ij')
    g = torch.stack([xx, yy], dim=-1).view(1, X, Y, 2)
    return g.expand(B, X, Y, 2)


def _make_grid_3d(B: int, X: int, Y: int, Z: int, device, dtype):
    gx = torch.linspace(0.0, 1.0, steps=X, device=device, dtype=dtype)
    gy = torch.linspace(0.0, 1.0, steps=Y, device=device, dtype=dtype)
    gz = torch.linspace(0.0, 1.0, steps=Z, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(gx, gy, gz, indexing='ij')
    g = torch.stack([xx, yy, zz], dim=-1).view(1, X, Y, Z, 3)
    return g.expand(B, X, Y, Z, 3)


class SGNOTorchApeWrapper(nn.Module):
    def __init__(self, core: nn.Module, spatial_dim: int, initial_step: int):
        super().__init__()
        self.core = core
        self.spatial_dim = int(spatial_dim)
        self.initial_step = int(initial_step)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        dev = u.device
        dtp = u.dtype

        if self.spatial_dim == 1:
            if u.ndim == 3:
                B, C, X = u.shape
                if self.initial_step != 1:
                    raise ValueError('When initial_step > 1, please provide history with shape B T C X or B T X C')
                base = u
                x = u.permute(0, 2, 1).contiguous()
            elif u.ndim == 4:
                B, T, C, X = u.shape
                if T != self.initial_step:
                    raise ValueError('The history length T must equal initial_step')
                base = u[:, -1].contiguous()
                x = u.permute(0, 3, 2, 1).contiguous().view(B, X, C * T)
            else:
                raise ValueError('For 1D input, only B C X, B T C X, or B T X C are supported')

            grid = _make_grid_1d(x.shape[0], x.shape[1], dev, dtp)
            delta = self.core(x, grid)
            delta = delta.squeeze(-2).permute(0, 2, 1).contiguous()
            delta = delta.to(dtype=base.dtype)
            return base + delta

        if self.spatial_dim == 2:
            if u.ndim == 4:
                B, C, X, Y = u.shape
                if self.initial_step != 1:
                    raise ValueError('When initial_step > 1, please provide history with shape B T C X Y or B T X Y C')
                base = u
                x = u.permute(0, 2, 3, 1).contiguous()
            elif u.ndim == 5:
                B, T, C, X, Y = u.shape
                if T != self.initial_step:
                    raise ValueError('The history length T must equal initial_step')
                base = u[:, -1].contiguous()
                x = u.permute(0, 3, 4, 2, 1).contiguous().view(B, X, Y, C * T)
            else:
                raise ValueError('For 2D input, only B C X Y, B T C X Y, or B T X Y C are supported')

            grid = _make_grid_2d(x.shape[0], x.shape[1], x.shape[2], dev, dtp)
            delta = self.core(x, grid)
            delta = delta.squeeze(-2).permute(0, 3, 1, 2).contiguous()
            delta = delta.to(dtype=base.dtype)
            return base + delta

        if self.spatial_dim == 3:
            if u.ndim == 5:
                B, C, X, Y, Z = u.shape
                if self.initial_step != 1:
                    raise ValueError(
                        'When initial_step > 1, please provide history with shape B T C X Y Z or B T X Y Z C'
                    )
                base = u
                x = u.permute(0, 2, 3, 4, 1).contiguous()
            elif u.ndim == 6:
                B, T, C, X, Y, Z = u.shape
                if T != self.initial_step:
                    raise ValueError('The history length T must equal initial_step')
                base = u[:, -1].contiguous()
                x = u.permute(0, 3, 4, 5, 2, 1).contiguous().view(B, X, Y, Z, C * T)
            else:
                raise ValueError('For 3D input, only B C X Y Z, B T C X Y Z, or B T X Y Z C are supported')

            grid = _make_grid_3d(x.shape[0], x.shape[1], x.shape[2], x.shape[3], dev, dtp)
            delta = self.core(x, grid)
            delta = delta.squeeze(-2).permute(0, 4, 1, 2, 3).contiguous()
            delta = delta.to(dtype=base.dtype)
            return base + delta

        raise ValueError('unsupported spatial_dim')


def build_sgno_from_config(
    network_config: str,
    num_spatial_dims: int,
    num_points: int,
    num_channels: int,
) -> nn.Module:
    cfg = _parse_network_config(network_config)

    width = int(cfg.get('width', 64))
    modes = int(cfg.get('modes', 16))
    n_blocks = int(cfg.get('n_blocks', 4))
    initial_step = int(cfg.get('initial_step', 1))
    dt = float(cfg.get('dt', 1.0))
    inner_steps = int(cfg.get('inner_steps', 1))

    use_beta = bool(cfg.get('use_beta', False))
    filter_type = str(cfg.get('filter_type', 'smooth'))
    filter_strength = float(cfg.get('filter_strength', 2.0))
    filter_order = int(cfg.get('filter_order', 8))
    padding = int(cfg.get('padding', 2))
    alpha_w = float(cfg.get('alpha_w', 1.0))
    alpha_g = float(cfg.get('alpha_g', 1.0))

    sd = int(num_spatial_dims)
    if sd == 1:
        core = SGNO1d(
            num_channels=num_channels,
            modes=modes,
            width=width,
            initial_step=initial_step,
            dt=dt,
            use_beta=use_beta,
            filter_type=filter_type,
            filter_strength=filter_strength,
            filter_order=filter_order,
            padding=padding,
            n_blocks=n_blocks,
            alpha_w=alpha_w,
            alpha_g=alpha_g,
            inner_steps=inner_steps,
        )
    elif sd == 2:
        core = SGNO2d(
            num_channels=num_channels,
            modes1=modes,
            modes2=modes,
            width=width,
            initial_step=initial_step,
            dt=dt,
            use_beta=use_beta,
            filter_type=filter_type,
            filter_strength=filter_strength,
            filter_order=filter_order,
            padding=padding,
            n_blocks=n_blocks,
            alpha_w=alpha_w,
            alpha_g=alpha_g,
            inner_steps=inner_steps,
        )
    elif sd == 3:
        core = SGNO3d(
            num_channels=num_channels,
            modes1=modes,
            modes2=modes,
            modes3=modes,
            width=width,
            initial_step=initial_step,
            dt=dt,
            use_beta=use_beta,
            filter_type=filter_type,
            filter_strength=filter_strength,
            filter_order=filter_order,
            padding=padding,
            n_blocks=n_blocks,
            alpha_w=alpha_w,
            alpha_g=alpha_g,
            inner_steps=inner_steps,
        )
    else:
        raise ValueError('num_spatial_dims must be 1 2 or 3')

    return SGNOTorchApeWrapper(core=core, spatial_dim=sd, initial_step=initial_step)
