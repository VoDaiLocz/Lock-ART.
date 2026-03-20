"""
Fast Gradient Sign Method (FGSM) Attack.

FGSM is a simple but effective single-step adversarial attack.
It computes the gradient of the loss with respect to the input image,
then adds a small perturbation in the direction that maximizes the loss.

Paper: "Explaining and Harnessing Adversarial Examples"
       https://arxiv.org/abs/1412.6572

Formula:
    adversarial = image + epsilon * sign(gradient)

Where:
    - epsilon: perturbation magnitude (e.g., 0.03 or 8/255)
    - gradient: ∂Loss/∂image
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralock.attacks.base import BaseAttack


class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    This is the simplest gradient-based attack. It's fast (single forward + backward pass)
    but less effective than iterative attacks like PGD.

    Usage:
        >>> from torchvision.models import resnet50, ResNet50_Weights
        >>> model = resnet50(weights=ResNet50_Weights.DEFAULT)
        >>> attack = FGSM(model, epsilon=0.03)
        >>> adversarial = attack(images, labels)

    For untargeted attack (fool the model without specific target):
        >>> adversarial = attack(images, labels, targeted=False)

    For targeted attack (fool the model into predicting specific class):
        >>> adversarial = attack(images, target_labels, targeted=True)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        device: str | torch.device | None = None,
    ):
        """
        Initialize FGSM attack.

        Args:
            model: Target classifier model.
                   Should accept (B, C, H, W) input and return (B, num_classes) logits.
            epsilon: Maximum L∞ perturbation. Default 0.03 ≈ 8/255.
                     Range guide:
                     - 0.01 (2.5/255): Very subtle, may not fool model
                     - 0.03 (8/255): Good balance of invisibility and effectiveness
                     - 0.06 (15/255): Strong attack, slightly visible
                     - 0.1 (25/255): Very strong, likely visible
            device: Computation device ('cpu', 'cuda', or torch.device)
        """
        super().__init__(model, epsilon, device)

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        targeted: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.

        Args:
            images: Input images of shape (B, C, H, W), values in [0, 1]
            labels: True labels for untargeted attack, target labels for targeted attack
                    Shape: (B,) containing class indices
            targeted: If True, perform targeted attack (fool into predicting labels)
                      If False, perform untargeted attack (fool away from labels)

        Returns:
            Adversarial images of shape (B, C, H, W), values in [0, 1]

        Example:
            >>> # Untargeted attack
            >>> adv_images = fgsm.generate(images, true_labels, targeted=False)

            >>> # Targeted attack
            >>> target = torch.zeros(batch_size, dtype=torch.long)  # Target class 0
            >>> adv_images = fgsm.generate(images, target, targeted=True)
        """
        # Move to device
        images = images.to(self.device)

        # If no labels provided, use model's predictions
        if labels is None:
            with torch.no_grad():
                outputs = self.model(images)
                labels = outputs.argmax(dim=1)

        labels = labels.to(self.device)

        # Enable gradient computation for input
        images_adv = images.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = self.model(images_adv)

        # Compute loss
        loss = F.cross_entropy(outputs, labels)

        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()

        # Get gradient sign
        grad_sign = images_adv.grad.sign()

        # Create perturbation
        if targeted:
            # For targeted attack: minimize loss (predict target class)
            perturbation = -self.epsilon * grad_sign
        else:
            # For untargeted attack: maximize loss (move away from true class)
            perturbation = self.epsilon * grad_sign

        # Apply perturbation
        adversarial = images + perturbation

        # Clamp to valid image range
        adversarial = self._clamp(adversarial)

        return adversarial.detach()

    def generate_with_info(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        targeted: bool = False,
    ) -> dict:
        """
        Generate adversarial examples and return detailed information.

        Returns:
            Dictionary containing:
            - adversarial: The adversarial images
            - perturbation: The added perturbation
            - original_preds: Model predictions on original images
            - adversarial_preds: Model predictions on adversarial images
            - success_rate: Percentage of successful attacks
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


def demo_fgsm():
    """
    Demonstrate FGSM attack on a pretrained model.

    This function shows how to use FGSM to create adversarial examples.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    from auralock.core.metrics import get_quality_report, print_quality_report

    print("🔬 FGSM Demo")
    print("-" * 40)

    # Load pretrained model
    print("Loading ResNet18...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Create FGSM attack
    attack = FGSM(model, epsilon=0.03)
    print(f"Attack config: epsilon={attack.epsilon}")

    # Create a sample image (random for demo)
    print("\nGenerating sample image...")
    images = torch.rand(1, 3, 224, 224)  # Random image

    # Generate adversarial example
    print("Generating adversarial example...")
    result = attack.generate_with_info(images)

    print("\n📊 Results:")
    print(f"   Original prediction:    class {result['original_preds'][0].item()}")
    print(f"   Adversarial prediction: class {result['adversarial_preds'][0].item()}")
    print(f"   Attack success rate:    {result['success_rate']*100:.1f}%")
    print(f"   Perturbation L2 norm:   {result['perturbation_l2']:.4f}")
    print(f"   Perturbation L∞ norm:   {result['perturbation_linf']:.4f}")

    # Quality report
    report = get_quality_report(images, result["adversarial"])
    print_quality_report(report)


if __name__ == "__main__":
    demo_fgsm()
