"""Unit tests for SIGReg implementation."""

import torch
from module import SIGReg


def test_sigreg_shape():
    """Verify SIGReg handles correct input shapes."""
    sigreg = SIGReg(knots=17, num_proj=1024)
    
    # Test different batch sizes and time steps
    test_cases = [
        (2, 64, 192),   # Small batch
        (4, 128, 192),  # Full batch size
        (8, 32, 192),   # Different T/B
    ]
    
    for T, B, D in test_cases:
        proj = torch.randn(T, B, D)
        loss = sigreg(proj.transpose(0, 1))
        
        assert loss.shape == torch.Size([]), f"Expected scalar loss, got {loss.shape}"
        assert torch.isfinite(loss).all(), "Loss should be finite"
        assert loss.item() >= 0, "SIGReg loss should be non-negative"


def test_sigreg_gradient_flow():
    """Verify SIGReg gradients flow properly."""
    sigreg = SIGReg(knots=17, num_proj=256)  # Reduce proj for speed
    
    z = torch.randn(4, 32, 192, requires_grad=True)
    loss = sigreg(z.transpose(0, 1))
    
    loss.backward()
    
    assert z.grad is not None, "Gradients should flow to input"
    assert torch.isfinite(z.grad).all(), "Graceints should be finite"
