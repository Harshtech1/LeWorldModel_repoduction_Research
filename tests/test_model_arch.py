"""Unit tests for LeWM model architecture and config alignment."""

import torch
from module import ConditionalBlock, SIGReg, MLP


def test_adaln_zero_init():
    """Verify AdaLN-zero modulation weights are initialized to zero."""
    blk = ConditionalBlock(dim=192, heads=4, dim_head=64, mlp_dim=2048)
    
    w_norm = blk.adaLN_modulation[-1].weight.norm().item()
    b_norm = blk.adaLN_modulation[-1].bias.norm().item()
    
    assert w_norm < 1e-6, f"AdaLN weight not zero-init: {w_norm}"
    assert b_norm < 1e-6, f"AdaLN bias not zero-init: {b_norm}"


def test_sigreg_normal_vs_collapsed():
    """Verify SIGReg returns low loss on Gaussian data and high on collapsed."""
    sigreg = SIGReg(knots=17, num_proj=256)  # Smaller proj for speed
    
    # Isotropic normal should have moderate SIGReg loss
    z_normal = torch.randn(4, 32, 192)
    loss_normal = sigreg(z_normal.transpose(0, 1)).item()
    
    # Collapsed tensor (all same vector) should have significantly higher loss
    z_collapsed = torch.zeros_like(z_normal)
    z_collapsed[:, :, 0] = 1.0
    
    loss_collapsed = sigreg(z_collapsed.transpose(0, 1)).item()
    
    diff = loss_collapsed - loss_normal
    assert diff > 0.3, \
        f"SIGReg should show clear separation: normal={loss_normal:.4f}, collapsed={loss_collapsed:.4f}, diff={diff:.4f}"


def test_mlp_batch_norm():
    """Verify MLP uses BatchNorm as specified."""
    mlp = MLP(input_dim=192, hidden_dim=2048, output_dim=192,
              norm_fn=lambda d: torch.nn.BatchNorm1d(d))
    
    # Check first layer is BatchNorm
    assert isinstance(mlp.net[1], torch.nn.BatchNorm1d), \
        f"Expected BatchNorm1d, got {type(mlp.net[1])}"
