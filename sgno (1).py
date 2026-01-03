from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import nn


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
    mesh = torch.meshgrid(*grids, indexing="ij")
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
        filter_type: str = "smooth",
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
            self.register_parameter("beta", None)

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
        if self.filter_type == "none":
            return None
        if self.filter_type in ("2/3", "two_thirds", "two-thirds"):
            return _two_thirds_mask_1d(self.modes1, device=device)
        return _smooth_filter_nd((self.modes1,), self.filter_strength, self.filter_order, device=device)

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, C, X = v.shape
        assert C == self.channels

        v_hat = torch.fft.rfft(v)
        g_hat = torch.fft.rfft(g)
        out_hat = torch.zeros_like(v_hat)

        m1 = min(self.modes1, v_hat.size(-1))

        lam = self._lambda(v.device)[:, :m1]
        z = (self.dt * lam).unsqueeze(0)
        expz = torch.exp(z)
        phi = _phi1(z)

        vh = v_hat[:, :, :m1]
        gh = g_hat[:, :, :m1]

        mix = self.mix[:, :, :m1].to(device=v.device)
        gmix = torch.einsum("bim,iom->bom", gh, mix)

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
        filter_type: str = "smooth",
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
            self.register_parameter("beta_pos", None)
            self.register_parameter("beta_neg", None)

        scale = 1.0 / (channels * channels)
        self.mix_pos = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, dtype=torch.cfloat))
        self.mix_neg = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, dtype=torch.cfloat))

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
        if self.filter_type == "none":
            return None
        if self.filter_type in ("2/3", "two_thirds", "two-thirds"):
            return _two_thirds_mask_2d(self.modes1, self.modes2, device=device)
        return _smooth_filter_nd((self.modes1, self.modes2), self.filter_strength, self.filter_order, device=device)

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, C, X, Y = v.shape
        assert C == self.channels

        v_hat = torch.fft.rfft2(v)
        g_hat = torch.fft.rfft2(g)
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
        gmp = torch.einsum("bixy,ioxy->boxy", gp, mixp)
        lin_p = exp_pos * vp
        forcing_p = (self.dt * phi_pos) * gmp
        if filt is not None:
            forcing_p = forcing_p * filt
        out_hat[:, :, :m1, :m2] = lin_p + forcing_p

        vn = v_hat[:, :, -m1:, :m2]
        gn = g_hat[:, :, -m1:, :m2]
        mixn = self.mix_neg[:, :, :m1, :m2].to(device=v.device)
        gmn = torch.einsum("bixy,ioxy->boxy", gn, mixn)
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
        filter_type: str = "smooth",
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
            self.register_parameter("beta1", None)
            self.register_parameter("beta2", None)
            self.register_parameter("beta3", None)
            self.register_parameter("beta4", None)

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
        if self.filter_type == "none":
            return None
        if self.filter_type in ("2/3", "two_thirds", "two-thirds"):
            return _two_thirds_mask_3d(self.modes1, self.modes2, self.modes3, device=device)
        return _smooth_filter_nd(
            (self.modes1, self.modes2, self.modes3), self.filter_strength, self.filter_order, device=device
        )

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = v.shape
        assert C == self.channels

        v_hat = torch.fft.rfftn(v, dim=[-3, -2, -1])
        g_hat = torch.fft.rfftn(g, dim=[-3, -2, -1])
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
            gmix = torch.einsum("bixyz,ioxyz->boxyz", gs, mix)
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
        filter_type: str = "smooth",
        filter_strength: float = 2.0,
        filter_order: int = 8,
        padding: int = 2,
    ):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.padding = padding

        self.fc0 = nn.Linear(initial_step * num_channels + 1, self.width)

        self.g0 = PointwiseMLP1d(self.width)
        self.g1 = PointwiseMLP1d(self.width)
        self.g2 = PointwiseMLP1d(self.width)
        self.g3 = PointwiseMLP1d(self.width)

        self.etd0 = SpectralETD1d(self.width, self.modes1, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd1 = SpectralETD1d(self.width, self.modes1, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd2 = SpectralETD1d(self.width, self.modes1, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd3 = SpectralETD1d(self.width, self.modes1, dt, use_beta, filter_type, filter_strength, filter_order)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x = F.pad(x, [0, self.padding])

        x = F.gelu(self.etd0(x, self.g0(x)) + self.w0(x))
        x = F.gelu(self.etd1(x, self.g1(x)) + self.w1(x))
        x = F.gelu(self.etd2(x, self.g2(x)) + self.w2(x))
        x = self.etd3(x, self.g3(x)) + self.w3(x)

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
        filter_type: str = "smooth",
        filter_strength: float = 2.0,
        filter_order: int = 8,
        padding: int = 2,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding

        self.fc0 = nn.Linear(initial_step * num_channels + 2, self.width)

        self.g0 = PointwiseMLP2d(self.width)
        self.g1 = PointwiseMLP2d(self.width)
        self.g2 = PointwiseMLP2d(self.width)
        self.g3 = PointwiseMLP2d(self.width)

        self.etd0 = SpectralETD2d(self.width, self.modes1, self.modes2, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd1 = SpectralETD2d(self.width, self.modes1, self.modes2, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd2 = SpectralETD2d(self.width, self.modes1, self.modes2, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd3 = SpectralETD2d(self.width, self.modes1, self.modes2, dt, use_beta, filter_type, filter_strength, filter_order)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = F.gelu(self.etd0(x, self.g0(x)) + self.w0(x))
        x = F.gelu(self.etd1(x, self.g1(x)) + self.w1(x))
        x = F.gelu(self.etd2(x, self.g2(x)) + self.w2(x))
        x = self.etd3(x, self.g3(x)) + self.w3(x)

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
        filter_type: str = "smooth",
        filter_strength: float = 2.0,
        filter_order: int = 8,
        padding: int = 6,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = padding

        self.fc0 = nn.Linear(initial_step * num_channels + 3, self.width)

        self.g0 = PointwiseMLP3d(self.width)
        self.g1 = PointwiseMLP3d(self.width)
        self.g2 = PointwiseMLP3d(self.width)
        self.g3 = PointwiseMLP3d(self.width)

        self.etd0 = SpectralETD3d(self.width, self.modes1, self.modes2, self.modes3, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd1 = SpectralETD3d(self.width, self.modes1, self.modes2, self.modes3, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd2 = SpectralETD3d(self.width, self.modes1, self.modes2, self.modes3, dt, use_beta, filter_type, filter_strength, filter_order)
        self.etd3 = SpectralETD3d(self.width, self.modes1, self.modes2, self.modes3, dt, use_beta, filter_type, filter_strength, filter_order)

        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = F.pad(x, [0, self.padding])

        x = F.gelu(self.etd0(x, self.g0(x)) + self.w0(x))
        x = F.gelu(self.etd1(x, self.g1(x)) + self.w1(x))
        x = F.gelu(self.etd2(x, self.g2(x)) + self.w2(x))
        x = self.etd3(x, self.g3(x)) + self.w3(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.unsqueeze(-2)
