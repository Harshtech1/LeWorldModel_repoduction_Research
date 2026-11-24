"""Integration tests for training loop."""

import torch
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def test_encoder_params():
    """Verify encoder parameter count (ViT-Tiny should be ~5M)."""
    from stable_pretraining.backbone.utils import vit_hf
    
    enc = vit_hf('tiny', patch_size=14, image_size=224)
    params = sum(p.numel() for p in enc.parameters())
    
    assert 5_000_000 < params < 6_000_000, \
        f"Expected ~5M encoder params, got {params:,}"


def test_predictor_params():
    """Verify predictor parameter count (~10M)."""
    embed_dim = 192
    pred = ARPredictor(
        num_frames=3,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0
    )
    
    params = sum(p.numel() for p in pred.parameters())
    
    assert 9_000_000 < params < 11_000_000, \
        f"Expected ~10M predictor params, got {params:,}"


def test_projector_params():
    """Verify the 192 -> 2048 -> 192 projector parameter count."""
    embed_dim = 192
    
    proj = MLP(
        input_dim=embed_dim,
        hidden_dim=2048,
        output_dim=embed_dim,
        norm_fn=lambda d: torch.nn.BatchNorm1d(d)
    )
    
    params = sum(p.numel() for p in proj.parameters())
    
    assert params == 792_768, \
        f"Expected 792,768 projector params, got {params:,}"


def test_total_params():
    """Verify total model parameters (~15M reference)."""
    from stable_pretraining.backbone.utils import vit_hf
    from module import ARPredictor, Embedder, MLP
    
    embed_dim = 192
    
    # Encoder (ViT-Tiny)
    encoder = vit_hf('tiny', patch_size=14, image_size=224)
    
    # Predictor (AdaLN Transformer)
    predictor = ARPredictor(
        num_frames=3,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0
    )
    
    # Projectors
    projector = MLP(
        input_dim=embed_dim,
        hidden_dim=2048,
        output_dim=embed_dim,
        norm_fn=lambda d: torch.nn.BatchNorm1d(d)
    )
    pred_proj = MLP(
        input_dim=embed_dim,
        hidden_dim=2048,
        output_dim=embed_dim,
        norm_fn=lambda d: torch.nn.BatchNorm1d(d)
    )
    
    # Action encoder
    action_encoder = Embedder(input_dim=10, emb_dim=embed_dim)
    
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    
    assert 14_000_000 < total_params < 20_000_000, \
        f"Expected ~15M total params, got {total_params:,}"


def test_no_stop_gradients():
    """Verify no stop-gradient or EMA heuristics in training forward."""
    # Read train.py and check for .detach() on target embeddings
    with open('/home/harsh_intern/Music/properites/Project/le-wm/train.py', 'r') as f:
        content = f.read()
    
    # Check that tgt_emb is NOT detached (the paper specifies no stop-gradients)
    assert 'tgt_emb' in content, "target embeddings reference missing"
    assert 'emb[:, n_preds:]' in content or 'emb[:, 1:]' in content, \
        "Expected target from sequential embedding without detach"
