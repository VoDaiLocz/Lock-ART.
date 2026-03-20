"""Protection profile presets and config resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ProtectionConfig:
    """Resolved attack configuration after applying a named profile."""

    profile: str
    method: str
    epsilon: float
    num_steps: int
    alpha: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert the config into a JSON-friendly payload."""
        return asdict(self)


PROFILE_PRESETS: dict[str, ProtectionConfig] = {
    "safe": ProtectionConfig(
        profile="safe",
        method="stylecloak",
        epsilon=0.01,
        num_steps=6,
        alpha=0.003,
    ),
    "balanced": ProtectionConfig(
        profile="balanced",
        method="stylecloak",
        epsilon=0.02,
        num_steps=12,
        alpha=0.003,
    ),
    "strong": ProtectionConfig(
        profile="strong",
        method="stylecloak",
        epsilon=0.032,
        num_steps=14,
        alpha=0.0053,
    ),
    "subject": ProtectionConfig(
        profile="subject",
        method="stylecloak",
        epsilon=0.05,
        num_steps=18,
        alpha=0.0055,
    ),
    "fortress": ProtectionConfig(
        profile="fortress",
        method="stylecloak",
        epsilon=0.06,
        num_steps=20,
        alpha=0.006,
    ),
    "blindfold": ProtectionConfig(
        profile="blindfold",
        method="blindfold",
        epsilon=0.09,
        num_steps=24,
        alpha=0.008,
    ),
}


def normalize_profile(profile: str) -> str:
    """Normalize user-provided profile names."""
    normalized = profile.strip().lower()
    aliases = {
        "default": "balanced",
        "medium": "balanced",
        "antidreambooth": "subject",
        "anti-dreambooth": "subject",
        "subject-strong": "subject",
        "max": "fortress",
        "maximum": "fortress",
        "blind": "blindfold",
        "obfuscate": "blindfold",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROFILE_PRESETS:
        valid_profiles = ", ".join(sorted(PROFILE_PRESETS))
        raise ValueError(
            f"Unknown profile '{profile}'. Valid profiles: {valid_profiles}."
        )
    return normalized


def resolve_protection_config(
    *,
    profile: str = "balanced",
    method: str | None = None,
    epsilon: float | None = None,
    num_steps: int | None = None,
    alpha: float | None = None,
) -> ProtectionConfig:
    """Resolve a final config by overlaying explicit values on a named profile."""
    resolved_profile = normalize_profile(profile)
    base = PROFILE_PRESETS[resolved_profile]
    return ProtectionConfig(
        profile=resolved_profile,
        method=method or base.method,
        epsilon=base.epsilon if epsilon is None else float(epsilon),
        num_steps=base.num_steps if num_steps is None else int(num_steps),
        alpha=base.alpha if alpha is None else float(alpha),
    )
