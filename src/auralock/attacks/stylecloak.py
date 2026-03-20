"""Style-aware robust attack for artwork protection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralock.attacks.base import BaseAttack
from auralock.core.style import (
    build_style_transform_suite,
    compute_embedding_distance,
    compute_style_distance,
    ensure_feature_bundle,
    gaussian_blur,
    high_frequency_energy,
)


class StyleCloak(BaseAttack):
    """Iterative feature-space attack tuned for style mimicry defense."""

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float | None = None,
        num_steps: int = 8,
        random_start: bool = False,
        style_weight: float = 1.0,
        embedding_weight: float = 1.0,
        pixel_weight: float = 0.3,
        high_frequency_weight: float = 0.45,
        worst_case_weight: float = 0.75,
        coarse_gradient_weight: float = 0.0,
        coarse_gradient_scale: float = 0.2,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(model, epsilon, device)

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if alpha is not None and alpha <= 0:
            raise ValueError("alpha must be positive.")
        if alpha is not None and alpha > epsilon:
            raise ValueError("alpha must be less than or equal to epsilon.")
        if any(weight < 0 for weight in (style_weight, embedding_weight)):
            raise ValueError("style and embedding weights must be non-negative.")
        if any(
            weight <= 0
            for weight in (pixel_weight, high_frequency_weight, worst_case_weight)
        ):
            raise ValueError("quality-preserving and robust weights must be positive.")
        if coarse_gradient_weight < 0:
            raise ValueError("coarse_gradient_weight must be non-negative.")
        if not 0.0 < coarse_gradient_scale <= 1.0:
            raise ValueError("coarse_gradient_scale must be between 0 and 1.")

        self.alpha = alpha if alpha is not None else epsilon / max(1, min(num_steps, 5))
        self.num_steps = num_steps
        self.random_start = random_start
        self.style_weight = style_weight
        self.embedding_weight = embedding_weight
        self.pixel_weight = pixel_weight
        self.high_frequency_weight = high_frequency_weight
        self.worst_case_weight = worst_case_weight
        self.coarse_gradient_weight = coarse_gradient_weight
        self.coarse_gradient_scale = coarse_gradient_scale

    def _objective(
        self,
        original_images: torch.Tensor,
        original_bundles: tuple[dict[str, torch.Tensor], ...],
        transforms: tuple[tuple[str, Any], ...],
        adversarial_images: torch.Tensor,
    ) -> torch.Tensor:
        transform_divergences: list[torch.Tensor] = []

        for original_bundle, (_, transform) in zip(
            original_bundles,
            transforms,
            strict=True,
        ):
            adversarial_bundle = ensure_feature_bundle(
                self.model,
                transform(adversarial_images),
            )
            style_divergence = compute_style_distance(
                original_bundle, adversarial_bundle
            )
            embedding_divergence = compute_embedding_distance(
                original_bundle,
                adversarial_bundle,
            )
            transform_divergences.append(
                self.style_weight * style_divergence
                + self.embedding_weight * embedding_divergence
            )

        divergence_stack = torch.stack(transform_divergences)
        robust_divergence = divergence_stack.mean() + self.worst_case_weight * (
            divergence_stack.min()
        )
        perturbation = adversarial_images - original_images
        pixel_penalty = F.mse_loss(adversarial_images, original_images)
        high_frequency_penalty = high_frequency_energy(
            perturbation,
            kernel_size=5,
            sigma=1.0,
        )
        low_frequency_alignment_bonus = F.mse_loss(
            gaussian_blur(adversarial_images),
            gaussian_blur(original_images),
        )

        return (
            robust_divergence
            - self.pixel_weight * pixel_penalty
            - self.high_frequency_weight * high_frequency_penalty
            + 0.1 * low_frequency_alignment_bonus
        )

    def generate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate style-aware adversarial examples."""
        del labels, kwargs

        images = images.to(self.device, dtype=torch.float32)
        original_images = images.detach().clone()
        transforms = build_style_transform_suite()

        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            adversarial_images = self._clamp(images + delta).detach()
        else:
            adversarial_images = images.detach().clone()

        with torch.no_grad():
            original_bundles = tuple(
                ensure_feature_bundle(self.model, transform(original_images))
                for _, transform in transforms
            )

        for _ in range(self.num_steps):
            adversarial_images = adversarial_images.detach().requires_grad_(True)
            objective = self._objective(
                original_images,
                original_bundles,
                transforms,
                adversarial_images,
            )

            self.model.zero_grad()
            objective.backward()
            gradient = adversarial_images.grad
            if gradient is None:
                raise RuntimeError("StyleCloak failed to compute input gradients.")

            update_direction = gradient
            if self.coarse_gradient_weight > 0:
                height, width = gradient.shape[-2:]
                coarse_height = max(
                    1,
                    int(round(height * self.coarse_gradient_scale)),
                )
                coarse_width = max(
                    1,
                    int(round(width * self.coarse_gradient_scale)),
                )
                coarse_gradient = F.interpolate(
                    gradient,
                    size=(coarse_height, coarse_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                coarse_gradient = F.interpolate(
                    coarse_gradient,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                update_direction = (
                    gradient + self.coarse_gradient_weight * coarse_gradient
                )

            adversarial_images = (
                adversarial_images + self.alpha * update_direction.sign()
            )
            perturbation = self._project(adversarial_images - original_images)
            adversarial_images = self._clamp(original_images + perturbation)

        return adversarial_images.detach()

    def generate_with_info(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Generate adversarial examples and expose perturbation details."""
        adversarial = self.generate(images, labels)
        perturbation = adversarial - images.to(self.device, dtype=torch.float32)
        transforms = build_style_transform_suite()
        with torch.no_grad():
            original_bundles = tuple(
                ensure_feature_bundle(
                    self.model,
                    transform(images.to(self.device, dtype=torch.float32)),
                )
                for _, transform in transforms
            )
        objective = float(
            self._objective(
                images.to(self.device, dtype=torch.float32),
                original_bundles,
                transforms,
                adversarial,
            )
            .detach()
            .item()
        )

        return {
            "adversarial": adversarial,
            "perturbation": perturbation,
            "objective": objective,
            "perturbation_l2": torch.norm(perturbation).item(),
            "perturbation_linf": torch.max(torch.abs(perturbation)).item(),
        }

    def get_info(self) -> dict[str, object]:
        info = super().get_info()
        info.update(
            {
                "alpha": self.alpha,
                "num_steps": self.num_steps,
                "random_start": self.random_start,
                "style_weight": self.style_weight,
                "embedding_weight": self.embedding_weight,
                "pixel_weight": self.pixel_weight,
                "high_frequency_weight": self.high_frequency_weight,
                "worst_case_weight": self.worst_case_weight,
                "coarse_gradient_weight": self.coarse_gradient_weight,
                "coarse_gradient_scale": self.coarse_gradient_scale,
            }
        )
        return info
