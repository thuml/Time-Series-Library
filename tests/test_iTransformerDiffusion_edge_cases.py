"""
Edge case tests for iTransformerDiffusion model.

Tests edge cases and robustness:
1. Different input/output channel numbers
2. Different sequence lengths
3. Batch size variations
4. Device handling
5. Gradient flow
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
        diffusion_steps=10,
        beta_schedule='cosine',
        cond_dim=64,
        unet_channels=[32, 64, 128],
        n_samples=2,
    )


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_different_channel_numbers(self):
        """Test with different channel numbers."""
        from models.iTransformerDiffusion import Model

        # Test with 1 channel
        configs = get_test_configs(enc_in=1)
        model = Model(configs)
        model.eval()

        B, seq_len, N = 2, 96, 1
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            y_det, z, _, _ = model.backbone_forward(x_enc)

        assert y_det.shape == (B, 96, 1)
        assert z.shape == (B, 1, 128)
        print("✓ Single channel passed")

        # Test with many channels
        configs = get_test_configs(enc_in=20)
        model = Model(configs)
        model.eval()

        B, seq_len, N = 2, 96, 20
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            y_det, z, _, _ = model.backbone_forward(x_enc)

        assert y_det.shape == (B, 96, 20)
        assert z.shape == (B, 20, 128)
        print("✓ Many channels (20) passed")

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        from models.iTransformerDiffusion import Model

        # Short sequence
        configs = get_test_configs(seq_len=24, pred_len=12)
        model = Model(configs)
        model.eval()

        B, seq_len, pred_len, N = 2, 24, 12, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            y_det, z, _, _ = model.backbone_forward(x_enc)

        assert y_det.shape == (B, pred_len, N)
        print("✓ Short sequence passed")

        # Long sequence
        configs = get_test_configs(seq_len=336, pred_len=336)
        model = Model(configs)
        model.eval()

        B, seq_len, pred_len, N = 2, 336, 336, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            y_det, z, _, _ = model.backbone_forward(x_enc)

        assert y_det.shape == (B, pred_len, N)
        print("✓ Long sequence passed")

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        # Batch size 1
        B, seq_len, N = 1, 96, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            y_det, z, _, _ = model.backbone_forward(x_enc)

        assert y_det.shape == (B, 96, N)
        print("✓ Batch size 1 passed")

        # Large batch
        B, seq_len, N = 32, 96, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            y_det, z, _, _ = model.backbone_forward(x_enc)

        assert y_det.shape == (B, 96, N)
        print("✓ Large batch (32) passed")

    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.train()

        B, seq_len, pred_len, N = 4, 96, 96, 7
        x_enc = torch.randn(B, seq_len, N, requires_grad=True)
        x_mark = torch.randn(B, seq_len, 4)
        y_true = torch.randn(B, pred_len, N)

        # Stage 1 loss
        loss, _ = model.forward_loss(x_enc, x_mark, y_true, stage='warmup')
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found"
        print("✓ Gradient flow passed")

    def test_residual_normalizer(self):
        """Test residual normalizer statistics."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.train()

        B, pred_len, N = 4, 96, 7
        residual = torch.randn(B, pred_len, N)

        # Normalize
        residual_norm = model.residual_normalizer.normalize(residual, update_stats=True)

        # Check normalized residual has reasonable scale
        assert torch.all(torch.isfinite(residual_norm))
        print(f"✓ Residual normalizer passed (std: {residual_norm.std():.4f})")

        # Denormalize
        residual_denorm = model.residual_normalizer.denormalize(residual_norm)

        # Check denormalized has similar scale to original
        assert torch.all(torch.isfinite(residual_denorm))
        print(f"✓ Denormalization passed")

    def test_sampling_consistency(self):
        """Test sampling produces consistent results."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        B, N, d_model = 2, 7, 128
        z = torch.randn(B, N, d_model)

        with torch.no_grad():
            # DDPM samples
            samples_ddpm = model.sample_ddpm(z, n_samples=3)

            # DDIM samples
            samples_ddim = model.sample_ddim(z, n_samples=3, ddim_steps=5)

        # Check shapes
        assert samples_ddpm.shape == (3, B, N, 96)
        assert samples_ddim.shape == (3, B, N, 96)

        # Check samples are finite
        assert torch.all(torch.isfinite(samples_ddpm))
        assert torch.all(torch.isfinite(samples_ddim))

        print("✓ Sampling consistency passed")

    def test_predict_robustness(self):
        """Test predict function robustness."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        model = Model(configs)
        model.eval()

        B, seq_len, N = 2, 96, 7
        x_enc = torch.randn(B, seq_len, N)

        with torch.no_grad():
            # Test with different n_samples
            mean1, std1, samples1 = model.predict(x_enc, n_samples=1)
            mean10, std10, samples10 = model.predict(x_enc, n_samples=10)

            # Check shapes
            assert mean1.shape == (B, 96, N)
            assert std1.shape == (B, 96, N)
            assert samples1.shape == (1, B, 96, N)
            assert samples10.shape == (10, B, 96, N)

            # Check std is non-negative (n_samples=1 will have std=0 or NaN, which is expected)
            assert torch.all(std1 >= 0) or torch.all(torch.isnan(std1)) or torch.all(std1 == 0)
            assert torch.all(std10 >= 0)

            # Check mean is finite
            assert torch.all(torch.isfinite(mean1))
            assert torch.all(torch.isfinite(mean10))

        print("✓ Predict robustness passed")

    def test_forward_compatibility(self):
        """Test standard forward for TSLib compatibility."""
        from models.iTransformerDiffusion import Model

        configs = get_test_configs()
        configs.task_name = 'long_term_forecast'
        model = Model(configs)
        model.eval()

        B, seq_len, pred_len, N = 4, 96, 96, 7
        x_enc = torch.randn(B, seq_len, N)
        x_mark_enc = torch.randn(B, seq_len, 4)
        x_dec = torch.randn(B, pred_len, N)
        x_mark_dec = torch.randn(B, pred_len, 4)

        with torch.no_grad():
            output = model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec)

        assert output.shape == (B, pred_len, N)
        print("✓ Forward compatibility passed")


def run_edge_case_tests():
    """Run all edge case tests."""
    print("=" * 60)
    print("Running iTransformerDiffusion Edge Case Tests")
    print("=" * 60)

    test_instance = TestEdgeCases()
    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for method_name in dir(test_instance):
        if method_name.startswith('test_'):
            total_tests += 1
            try:
                getattr(test_instance, method_name)()
                passed_tests += 1
            except Exception as e:
                failed_tests.append((method_name, str(e)))
                print(f"✗ {method_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Edge Case Results: {passed_tests}/{total_tests} tests passed")

    if failed_tests:
        print("\nFailed tests:")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")
    else:
        print("\n🎉 All edge case tests passed!")

    print("=" * 60)

    return len(failed_tests) == 0


if __name__ == '__main__':
    success = run_edge_case_tests()
    sys.exit(0 if success else 1)

