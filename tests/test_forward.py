import torch
from sgno import build_sgno_from_config

def test_forward_1d():
    cfg = "sgno;width=8;modes=8;n_blocks=1;initial_step=1;dt=1.0;inner_steps=1;use_beta=false;filter_type=none;filter_strength=0.0;filter_order=8;padding=2;alpha_w=1.0;alpha_g=1.0"
    m = build_sgno_from_config(cfg, 1, 64, 1)
    u = torch.randn(2, 1, 64)
    y = m(u)
    assert y.shape == u.shape

def test_forward_2d():
    cfg = "sgno;width=8;modes=4;n_blocks=1;initial_step=1;dt=1.0;inner_steps=1;use_beta=false;filter_type=smooth;filter_strength=1.0;filter_order=8;padding=2;alpha_w=1.0;alpha_g=1.0"
    m = build_sgno_from_config(cfg, 2, 16, 1)
    u = torch.randn(2, 1, 16, 16)
    y = m(u)
    assert y.shape == u.shape
