"""
Projected Gradient Descent (PGD) Attack.

PGD is an iterative version of FGSM that applies multiple smaller steps
and projects the result back to the epsilon-ball after each step.

Paper: "Towards Deep Learning Models Resistant to Adversarial Attacks"
       https://arxiv.org/abs/1706.06083

PGD is considered one of the strongest first-order attacks and is often
used as a benchmark for adversarial robustness evaluation.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralock.attacks.base import BaseAttack


class PGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) adversarial attack.

    PGD is stronger than FGSM because it uses multiple iterations,
    each with a smaller step size, and includes random initialization.

    Args:
        model: Target classifier model
        epsilon: Maximum L∞ perturbation (total budget)
        alpha: Step size per iteration (default: epsilon/4)
        num_steps: Number of iterations (default: 10)
        random_start: Whether to start from random point in epsilon-ball

    Usage:
        >>> attack = PGD(model, epsilon=0.03, num_steps=10)
        >>> adversarial = attack(images, labels)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float | None = None,
        num_steps: int = 10,
        random_start: bool = True,
        device: str | torch.device | None = None,
    ):
        super().__init__(model, epsilon, device)

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if alpha is not None and alpha <= 0:
            raise ValueError("alpha must be positive.")
        if alpha is not None and alpha > epsilon:
            raise ValueError("alpha must be less than or equal to epsilon.")

        # Default step size: epsilon / 4
        self.alpha = alpha if alpha is not None else epsilon / 4
        self.num_steps = num_steps
        self.random_start = random_start

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        targeted: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.

        Args:
            images: Input images of shape (B, C, H, W), values in [0, 1]
            labels: True labels (untargeted) or target labels (targeted)
            targeted: If True, minimize loss toward target class

        Returns:
            Adversarial images of shape (B, C, H, W), values in [0, 1]
        """
        images = images.to(self.device)

        # Get labels if not provided
        if labels is None:
            with torch.no_grad():
                outputs = self.model(images)
                labels = outputs.argmax(dim=1)

        labels = labels.to(self.device)

        # Initialize adversarial images
        if self.random_start:
            # Start from random point within epsilon-ball
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(images + delta, 0.0, 1.0)
        else:
            adv_images = images.clone()

        # Iterative attack
        for _ in range(self.num_steps):
            adv_images = adv_images.detach().requires_grad_(True)

            # Forward pass
            outputs = self.model(adv_images)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Get gradient sign
            grad_sign = adv_images.grad.sign()

            # Update adversarial images
            if targeted:
                adv_images = adv_images - self.alpha * grad_sign
            else:
                adv_images = adv_images + self.alpha * grad_sign

            # Project back to epsilon-ball around original image
            perturbation = adv_images - images
            perturbation = self._project(perturbation)
            adv_images = images + perturbation

            # Clamp to valid image range
            adv_images = self._clamp(adv_images)

        return adv_images.detach()

    def generate_with_info(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        targeted: bool = False,
    ) -> dict:
        """
        Generate adversarial examples and return detailed information.
        """
        images = images.to(self.device)

        # Get original predictions
        with torch.no_grad():
            original_outputs = self.model(images)
            original_preds = original_outputs.argmax(dim=1)

        if labels is None:
            labels = original_preds
        labels = labels.to(self.device)

        # Generate adversarial examples
        adversarial = self.generate(images, labels, targeted)

        # Get adversarial predictions
        with torch.no_grad():
            adv_outputs = self.model(adversarial)
            adv_preds = adv_outputs.argmax(dim=1)

        # Calculate success rate
        if targeted:
            success = (adv_preds == labels).float().mean().item()
        else:
            success = (adv_preds != labels).float().mean().item()

        perturbation = adversarial - images

        return {
            "adversarial": adversarial,
            "perturbation": perturbation,
            "original_preds": original_preds.cpu(),
            "adversarial_preds": adv_preds.cpu(),
            "success_rate": success,
            "perturbation_l2": torch.norm(perturbation).item(),
            "perturbation_linf": torch.max(torch.abs(perturbation)).item(),
        }

    def get_info(self) -> dict:
        """Return attack configuration."""
        info = super().get_info()
        info.update(
            {
                "alpha": self.alpha,
                "num_steps": self.num_steps,
                "random_start": self.random_start,
            }
        )
        return info


def demo_pgd():
    """Demonstrate PGD attack."""
    from torchvision.models import ResNet18_Weights, resnet18

    from auralock.core.metrics import get_quality_report

    print("🔬 PGD Demo")
    print("-" * 40)

    # Load model
    print("Loading ResNet18...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Create attacks
    fgsm_attack = __import__("auralock.attacks.fgsm", fromlist=["FGSM"]).FGSM(
        model, epsilon=0.03
    )
    pgd_attack = PGD(model, epsilon=0.03, num_steps=10)

    # Test image
    images = torch.rand(1, 3, 224, 224)

    print("\nComparing FGSM vs PGD:")
    print("-" * 40)

    # FGSM
    fgsm_result = fgsm_attack.generate_with_info(images)
    print(
        f"FGSM: Success={fgsm_result['success_rate']*100:.0f}%, L∞={fgsm_result['perturbation_linf']:.4f}"
    )

    # PGD
    pgd_result = pgd_attack.generate_with_info(images)
    print(
        f"PGD:  Success={pgd_result['success_rate']*100:.0f}%, L∞={pgd_result['perturbation_linf']:.4f}"
    )

    # Quality comparison
    fgsm_report = get_quality_report(images, fgsm_result["adversarial"])
    pgd_report = get_quality_report(images, pgd_result["adversarial"])

    print("\nQuality Comparison:")
    print(f"FGSM: PSNR={fgsm_report['psnr_db']:.1f}dB, SSIM={fgsm_report['ssim']:.4f}")
    print(f"PGD:  PSNR={pgd_report['psnr_db']:.1f}dB, SSIM={pgd_report['ssim']:.4f}")


if __name__ == "__main__":
    demo_pgd()
