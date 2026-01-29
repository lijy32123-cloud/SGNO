import torch
from sgno import build_sgno_from_config

def _run_1d():
    cfg = "sgno;width=8;modes=8;n_blocks=1;initial_step=1;dt=1.0;inner_steps=1;use_beta=false;filter_type=none;filter_strength=0.0;filter_order=8;padding=2;alpha_w=1.0;alpha_g=1.0"
    m = build_sgno_from_config(cfg, 1, 128, 1)
    u = torch.randn(2, 1, 128)
    y = m(u)
    assert y.shape == u.shape

def _run_2d():
    cfg = "sgno;width=8;modes=4;n_blocks=1;initial_step=1;dt=1.0;inner_steps=1;use_beta=false;filter_type=smooth;filter_strength=1.0;filter_order=8;padding=2;alpha_w=1.0;alpha_g=1.0"
    m = build_sgno_from_config(cfg, 2, 32, 1)
    u = torch.randn(2, 1, 32, 32)
    y = m(u)
    assert y.shape == u.shape

def _run_3d():
    cfg = "sgno;width=4;modes=4;n_blocks=1;initial_step=1;dt=1.0;inner_steps=1;use_beta=false;filter_type=none;filter_strength=0.0;filter_order=8;padding=2;alpha_w=1.0;alpha_g=1.0"
    m = build_sgno_from_config(cfg, 3, 16, 1)
    u = torch.randn(1, 1, 16, 16, 16)
    y = m(u)
    assert y.shape == u.shape

def main():
    torch.set_grad_enabled(False)
    _run_1d()
    _run_2d()
    _run_3d()
    print("ok")

if __name__ == "__main__":
    main()
