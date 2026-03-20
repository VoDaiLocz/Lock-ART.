"""
Adversarial attack implementations for image protection.

Available attacks:
- FGSM (Fast Gradient Sign Method): Fast, single-step attack
- PGD (Projected Gradient Descent): Stronger, iterative attack
- StyleCloak: Style-aware robust protection against mimicry pipelines
"""

from auralock.attacks.base import BaseAttack
from auralock.attacks.fgsm import FGSM
from auralock.attacks.pgd import PGD
from auralock.attacks.stylecloak import StyleCloak

__all__ = [
    "BaseAttack",
    "FGSM",
    "PGD",
    "StyleCloak",
]
