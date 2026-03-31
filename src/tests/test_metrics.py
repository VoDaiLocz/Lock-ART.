"""Tests for quality metrics."""

import numpy as np
import torch


class TestPSNR:
    """Tests for PSNR calculation."""

    def test_psnr_identical_images(self):
        """PSNR of identical images should be very high."""
        from auralock.core.metrics import calculate_psnr

        img = torch.rand(3, 100, 100)
        psnr = calculate_psnr(img, img)

        # Identical images should have inf PSNR, but due to float precision
        # it might be a very large number
        assert psnr > 50

    def test_psnr_with_noise(self):
        """PSNR should decrease as noise increases."""
        from auralock.core.metrics import calculate_psnr

        original = torch.rand(3, 100, 100)

        psnrs = []
        for noise_level in [0.01, 0.05, 0.1]:
            noisy = torch.clamp(
                original + torch.randn_like(original) * noise_level, 0, 1
            )
            psnr = calculate_psnr(original, noisy)
            psnrs.append(psnr)

        # PSNR should generally decrease with more noise
        # (not strictly monotonic due to random noise)
        assert psnrs[0] > psnrs[2]

    def test_psnr_accepts_numpy(self):
        """PSNR should work with numpy arrays."""
        from auralock.core.metrics import calculate_psnr

        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = np.random.rand(100, 100, 3).astype(np.float32)

        psnr = calculate_psnr(img1, img2)

        assert isinstance(psnr, float)
        assert psnr > 0


class TestSSIM:
    """Tests for SSIM calculation."""

    def test_ssim_identical_images(self):
        """SSIM of identical images should be 1.0."""
        from auralock.core.metrics import calculate_ssim

        img = torch.rand(3, 100, 100)
        ssim = calculate_ssim(img, img)

        assert ssim > 0.99

    def test_ssim_range(self):
        """SSIM should be between -1 and 1."""
        from auralock.core.metrics import calculate_ssim

        img1 = torch.rand(3, 100, 100)
        img2 = torch.rand(3, 100, 100)

        ssim = calculate_ssim(img1, img2)

        assert -1 <= ssim <= 1

    def test_ssim_with_noise(self):
        """SSIM should decrease with more noise."""
        from auralock.core.metrics import calculate_ssim

        original = torch.rand(3, 100, 100)

        # Small noise
        small_noise = original + torch.randn_like(original) * 0.01
        small_noise = torch.clamp(small_noise, 0, 1)
        ssim_small = calculate_ssim(original, small_noise)

        # Large noise
        large_noise = original + torch.randn_like(original) * 0.2
        large_noise = torch.clamp(large_noise, 0, 1)
        ssim_large = calculate_ssim(original, large_noise)

        assert ssim_small > ssim_large


class TestQualityReport:
    """Tests for quality report generation."""

    def test_get_quality_report_keys(self):
        """Test that quality report has expected keys."""
        from auralock.core.metrics import get_quality_report

        img1 = torch.rand(3, 100, 100)
        img2 = torch.rand(3, 100, 100)

        report = get_quality_report(img1, img2)

        assert "psnr_db" in report
        assert "ssim" in report
        assert "l2_distance" in report
        assert "linf_distance" in report
        assert "overall_quality" in report

    def test_quality_excellent(self):
        """Test that near-identical images get Excellent rating."""
        from auralock.core.metrics import get_quality_report

        img = torch.rand(3, 100, 100)
        similar = img + torch.randn_like(img) * 0.001
        similar = torch.clamp(similar, 0, 1)

        report = get_quality_report(img, similar)

        assert report["overall_quality"] in ["Excellent", "Good"]

    def test_quality_poor(self):
        """Test that very different images get Poor rating."""
        from auralock.core.metrics import get_quality_report

        img1 = torch.zeros(3, 100, 100)
        img2 = torch.ones(3, 100, 100)

        report = get_quality_report(img1, img2)

        assert report["overall_quality"] == "Poor"


class TestProtectionReadabilityReport:
    """Tests for protection readability report with warning metadata."""

    def test_protection_report_includes_warnings(self):
        """Test that protection report includes warning metadata."""
        from auralock.core.metrics import (
            PROTECTION_SCORE_DISCLAIMER,
            PROTECTION_SCORE_METRIC_TYPE,
            PROTECTION_SCORE_WARNING,
            get_protection_readability_report,
        )

        original = torch.rand(1, 3, 100, 100)
        modified = original + torch.randn_like(original) * 0.1
        modified = torch.clamp(modified, 0, 1)

        report = get_protection_readability_report(original, modified)

        # Check that warning metadata is present
        assert "warning" in report
        assert "metric_type" in report
        assert "disclaimer" in report

        # Check that values match constants
        assert report["warning"] == PROTECTION_SCORE_WARNING
        assert report["metric_type"] == PROTECTION_SCORE_METRIC_TYPE
        assert report["disclaimer"] == PROTECTION_SCORE_DISCLAIMER

    def test_protection_report_assessment_includes_proxy_space(self):
        """Test that assessment labels include '(proxy space)' suffix."""
        from auralock.core.metrics import get_protection_readability_report

        original = torch.rand(1, 3, 100, 100)
        modified = original + torch.randn_like(original) * 0.1
        modified = torch.clamp(modified, 0, 1)

        report = get_protection_readability_report(original, modified)

        # Check that assessment includes proxy space indicator
        assert "(proxy space)" in report["assessment"]

    def test_protection_report_has_all_expected_keys(self):
        """Test that protection report has all expected keys including new warning fields."""
        from auralock.core.metrics import get_protection_readability_report

        original = torch.rand(1, 3, 100, 100)
        modified = original + torch.randn_like(original) * 0.1
        modified = torch.clamp(modified, 0, 1)

        report = get_protection_readability_report(original, modified)

        # Original keys
        assert "style_similarity" in report
        assert "embedding_similarity" in report
        assert "robust_style_similarity" in report
        assert "robust_embedding_similarity" in report
        assert "protection_score" in report
        assert "assessment" in report
        assert "transform_style_similarities" in report
        assert "transform_embedding_similarities" in report

        # New warning keys
        assert "metric_type" in report
        assert "warning" in report
        assert "disclaimer" in report
