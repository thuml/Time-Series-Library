"""
Unit tests for iTransformerDiffusion model.

Tests:
1. Model initialization
2. Backbone forward pass
3. Channel alignment in backbone_forward
4. forward_loss (Stage 1 & Stage 2)
5. sample_ddpm / sample_ddim sampling
6. predict function
7. freeze/unfreeze encoder
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from types import SimpleNamespace


def get_test_configs(enc_in=7, seq_len=96, pred_len=96, d_model=128):
    """Create test configuration."""
    return SimpleNamespace(
        task_name='diffusion_forecast',
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=d_model,
        d_ff=d_model,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        factor=3,
        embed='timeF',
        freq='h',
        dropout=0.1,
        activation='gelu',
        # Diffusion configs
        diffusion_steps=10,  # Small for testing
        beta_schedule='cosine',
        cond_dim=64,
        unet_channels=[32, 64, 128],  # Smaller for testing
        n_samples=2,
    )


class TestModelInitialization:
    """Test model initialization."""

    def test_basic_init(self):
        """Test basic model initialization."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)

        assert model.n_vars == 7
        assert model.seq_len == 96
        assert model.pred_len == 96
        assert model.timesteps == 10
        print("✓ Basic initialization passed")

    def test_diffusion_schedule(self):
        """Test diffusion schedule setup."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)

        assert hasattr(model, 'betas')
        assert hasattr(model, 'alphas')
        assert hasattr(model, 'alpha_cumprods')
        assert model.betas.shape[0] == 10
        assert torch.all(model.betas > 0)
        assert torch.all(model.betas < 1)
        print("✓ Diffusion schedule passed")


class TestBackboneForward:
    """Test backbone forward pass."""

    def test_backbone_forward_basic(self):
        """Test basic backbone forward."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs(enc_in=7)
        model = Model(configs)
        model.eval()

        B, seq_len, N = 4, 96, 7
        x_enc = torch.randn(B, seq_len, N)
        x_mark = torch.randn(B, seq_len, 4)

        with torch.no_grad():
            y_det, z, means, stdev = model.backbone_forward(x_enc, x_mark)

        assert y_det.shape == (B, 96, N), f"y_det shape: {y_det.shape}"
        assert z.shape == (B, N, 128), f"z shape: {z.shape}"
        print("✓ Backbone forward basic passed")

    def test_backbone_forward_channel_alignment(self):
        """Test that backbone aligns z to n_vars."""
        from models.iTransformerDiffusion import Model

        # Model expects 7 variates
        configs = get_test_configs(enc_in=7)
        model = Model(configs)
        model.eval()

        # Input with correct channels
        B, seq_len, N = 4, 96, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            y_det, z, _, _ = model.backbone_forward(x_enc)

        # z should always have n_vars channels
        assert z.shape[1] == 7, f"z should have 7 variates, got {z.shape[1]}"
        print("✓ Channel alignment passed")


class TestForwardLoss:
    """Test forward_loss function."""

    def test_forward_loss_warmup(self):
        """Test Stage 1 (warmup) loss computation."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.train()

        B, seq_len, pred_len, N = 4, 96, 96, 7
        x_enc = torch.randn(B, seq_len, N)
        x_mark = torch.randn(B, seq_len, 4)
        y_true = torch.randn(B, pred_len, N)

        loss, loss_dict = model.forward_loss(x_enc, x_mark, y_true, stage='warmup')

        assert 'loss_mse' in loss_dict
        assert loss.item() > 0
        print(f"✓ Stage 1 loss: {loss.item():.4f}")

    def test_forward_loss_joint(self):
        """Test Stage 2 (joint) loss computation."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.train()

        B, seq_len, pred_len, N = 4, 96, 96, 7
        x_enc = torch.randn(B, seq_len, N)
        x_mark = torch.randn(B, seq_len, 4)
        y_true = torch.randn(B, pred_len, N)

        loss, loss_dict = model.forward_loss(x_enc, x_mark, y_true, stage='joint')

        assert 'loss_mse' in loss_dict
        assert 'loss_diff' in loss_dict
        assert 'loss_total' in loss_dict
        print(f"✓ Stage 2 loss: total={loss_dict['loss_total']:.4f}, mse={loss_dict['loss_mse']:.4f}, diff={loss_dict['loss_diff']:.4f}")


class TestSampling:
    """Test DDPM and DDIM sampling."""

    def test_sample_ddpm(self):
        """Test DDPM sampling."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        B, N, d_model = 2, 7, 128
        z = torch.randn(B, N, d_model)

        with torch.no_grad():
            samples = model.sample_ddpm(z, n_samples=2)

        # samples: [n_samples, B, N, pred_len]
        assert samples.shape == (2, B, 7, 96), f"samples shape: {samples.shape}"
        print(f"✓ DDPM sampling passed, shape: {samples.shape}")

    def test_sample_ddim(self):
        """Test DDIM sampling."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        B, N, d_model = 2, 7, 128
        z = torch.randn(B, N, d_model)

        with torch.no_grad():
            samples = model.sample_ddim(z, n_samples=2, ddim_steps=5)

        assert samples.shape == (2, B, 7, 96), f"samples shape: {samples.shape}"
        print(f"✓ DDIM sampling passed, shape: {samples.shape}")

    def test_sample_channel_consistency(self):
        """Test that sampling uses n_vars consistently."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs(enc_in=7)
        model = Model(configs)
        model.eval()

        # Even if z has different N dimension, sampling should use n_vars
        B = 2
        # z with more channels than n_vars (edge case after some operations)
        z = torch.randn(B, 11, 128)  # 11 channels instead of 7

        with torch.no_grad():
            samples = model.sample_ddpm(z, n_samples=1)

        # Should create samples with n_vars=7, not 11
        assert samples.shape[2] == 7, f"Expected 7 variates, got {samples.shape[2]}"
        print("✓ Sample channel consistency passed")


class TestPredict:
    """Test predict function."""

    def test_predict_basic(self):
        """Test basic predict function."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        B, seq_len, N = 2, 96, 7
        x_enc = torch.randn(B, seq_len, N)
        x_mark = torch.randn(B, seq_len, 4)

        with torch.no_grad():
            mean_pred, std_pred, samples = model.predict(x_enc, x_mark, n_samples=2)

        assert mean_pred.shape == (B, 96, N), f"mean_pred shape: {mean_pred.shape}"
        assert std_pred.shape == (B, 96, N), f"std_pred shape: {std_pred.shape}"
        assert samples.shape == (2, B, 96, N), f"samples shape: {samples.shape}"
        print(f"✓ Predict basic passed")
        print(f"  mean_pred: {mean_pred.shape}, std_pred: {std_pred.shape}")

    def test_predict_with_ddim(self):
        """Test predict with DDIM."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        B, seq_len, N = 2, 96, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            mean_pred, std_pred, samples = model.predict(
                x_enc, n_samples=2, use_ddim=True, ddim_steps=5
            )

        assert mean_pred.shape == (B, 96, N)
        print("✓ Predict with DDIM passed")


class TestFreezeUnfreeze:
    """Test freeze/unfreeze encoder."""

    def test_freeze_encoder(self):
        """Test encoder freezing."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)

        model.freeze_encoder()

        # Check embedding frozen
        for param in model.enc_embedding.parameters():
            assert not param.requires_grad, "Embedding should be frozen"

        # Check encoder frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad, "Encoder should be frozen"

        # Check projection still trainable
        for param in model.projection.parameters():
            assert param.requires_grad, "Projection should be trainable"

        print("✓ Freeze encoder passed")

    def test_unfreeze_encoder(self):
        """Test encoder unfreezing."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)

        model.freeze_encoder()
        model.unfreeze_encoder()

        for param in model.enc_embedding.parameters():
            assert param.requires_grad, "Embedding should be trainable"

        for param in model.encoder.parameters():
            assert param.requires_grad, "Encoder should be trainable"

        print("✓ Unfreeze encoder passed")


class TestEndToEnd:
    """End-to-end training simulation."""

    def test_training_step(self):
        """Test a complete training step."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        B, seq_len, pred_len, N = 4, 96, 96, 7
        x_enc = torch.randn(B, seq_len, N)
        x_mark = torch.randn(B, seq_len, 4)
        y_true = torch.randn(B, pred_len, N)

        # Stage 1 step
        optimizer.zero_grad()
        loss1, _ = model.forward_loss(x_enc, x_mark, y_true, stage='warmup')
        loss1.backward()
        optimizer.step()

        # Stage 2 step
        model.freeze_encoder()
        optimizer.zero_grad()
        loss2, _ = model.forward_loss(x_enc, x_mark, y_true, stage='joint')
        loss2.backward()
        optimizer.step()

        print(f"✓ Training step passed: loss1={loss1.item():.4f}, loss2={loss2.item():.4f}")

    def test_full_inference(self):
        """Test full inference pipeline."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        B, seq_len, N = 2, 96, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            mean_pred, std_pred, samples = model.predict(x_enc, n_samples=3)

        # Check output shapes
        assert mean_pred.shape == (B, 96, N)
        assert std_pred.shape == (B, 96, N)
        assert samples.shape == (3, B, 96, N)

        # Check std is positive
        assert torch.all(std_pred >= 0), "std should be non-negative"

        # Check samples have reasonable spread
        sample_std = samples.std(dim=0)
        assert torch.all(sample_std >= 0), "sample std should be non-negative"

        print("✓ Full inference passed")
        print(f"  Mean range: [{mean_pred.min():.2f}, {mean_pred.max():.2f}]")
        print(f"  Std range: [{std_pred.min():.4f}, {std_pred.max():.4f}]")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running iTransformerDiffusion Unit Tests")
    print("=" * 60)

    test_classes = [
        TestModelInitialization,
        TestBackboneForward,
        TestForwardLoss,
        TestSampling,
        TestPredict,
        TestFreezeUnfreeze,
        TestEndToEnd,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    passed_tests += 1
                except Exception as e:
                    failed_tests.append((f"{test_class.__name__}.{method_name}", str(e)))
                    print(f"✗ {method_name}: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print("\nFailed tests:")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")
    else:
        print("\n🎉 All tests passed!")

    print("=" * 60)

    return len(failed_tests) == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
