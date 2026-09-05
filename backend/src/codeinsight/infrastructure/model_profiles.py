"""截图价格与能力约束对应的版本化模型注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

PRICE_VERSION = "frontier-stars-2026-09-04"
DEFAULT_MODEL_ID = "deepseek-v4-flash"


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    provider: str
    input_price_per_million: Decimal
    output_price_per_million: Decimal
    cached_input_price_per_million: Decimal
    price_version: str
    capabilities: frozenset[str]
    quality_tier: str
    context_window_tokens: int = 1_000_000
    timeout_seconds: float = 60.0
    fallback_eligible: bool = False

    @property
    def nominal_price(self) -> Decimal:
        return self.input_price_per_million + self.output_price_per_million


@dataclass(frozen=True)
class RouteProfile:
    scene: str
    primary_model_id: str
    fallback_model_ids: tuple[str, ...]
    required_capabilities: frozenset[str]


class ModelRegistry:
    def __init__(self, profiles: tuple[ModelProfile, ...]) -> None:
        self._profiles = {profile.model_id: profile for profile in profiles}
        if len(self._profiles) != len(profiles):
            raise ValueError("model_id 必须唯一")

    def require(self, model_id: str) -> ModelProfile:
        try:
            return self._profiles[model_id]
        except KeyError:
            raise KeyError(f"未登记模型：{model_id}") from None

    def all(self) -> tuple[ModelProfile, ...]:
        return tuple(self._profiles.values())

    def cheaper_capable_fallbacks(
        self, primary_model_id: str, required: frozenset[str]
    ) -> tuple[str, ...]:
        primary = self.require(primary_model_id)
        candidates = [
            profile
            for profile in self._profiles.values()
            if profile.model_id != primary_model_id
            and profile.fallback_eligible
            and required <= profile.capabilities
            and profile.nominal_price < primary.nominal_price
        ]
        candidates.sort(key=lambda item: (item.nominal_price, item.model_id))
        return tuple(item.model_id for item in candidates)


def _profile(
    model_id: str,
    input_price: str,
    output_price: str,
    cached_price: str,
    capabilities: set[str],
    *,
    quality_tier: str = "candidate",
    fallback_eligible: bool = False,
) -> ModelProfile:
    return ModelProfile(
        model_id=model_id,
        provider="frontier-openai-compatible",
        input_price_per_million=Decimal(input_price),
        output_price_per_million=Decimal(output_price),
        cached_input_price_per_million=Decimal(cached_price),
        price_version=PRICE_VERSION,
        capabilities=frozenset(capabilities),
        quality_tier=quality_tier,
        context_window_tokens=1_000_000,
        timeout_seconds=60.0,
        fallback_eligible=fallback_eligible,
    )


def default_model_registry() -> ModelRegistry:
    """登记图片中可见的 12 个模型；未明确能力的模型不进入自动降级链。"""
    return ModelRegistry(
        (
            _profile("codex-auto-review", "1.75", "14", "0.175", {"text"}),
            _profile(
                DEFAULT_MODEL_ID,
                "12",
                "36",
                "0.3996",
                {"text", "tools", "structured_output"},
                quality_tier="best",
            ),
            _profile(
                "deepseek-v4-flash-vision-exp",
                "12",
                "36",
                "0.3996",
                {"text", "vision", "reasoning", "tools", "structured_output"},
            ),
            _profile("deepseek-v4-pro", "36", "108", "1.1988", {"text"}),
            _profile(
                "glm-5.2",
                "30",
                "105",
                "7.5",
                {"text", "reasoning", "tools", "structured_output"},
            ),
            _profile(
                "glm-5.2-fast-preview",
                "60",
                "210",
                "15",
                {"text", "reasoning", "tools", "structured_output"},
            ),
            _profile(
                "gpt-5.4",
                "2.5",
                "15",
                "0.25",
                {"text", "reasoning", "tools", "structured_output"},
                fallback_eligible=True,
            ),
            _profile(
                "gpt-5.4-mini",
                "0.75",
                "4.5",
                "0.075",
                {"text", "reasoning", "tools", "structured_output"},
                fallback_eligible=True,
            ),
            _profile(
                "gpt-5.5",
                "5",
                "30",
                "0.5",
                {"text", "reasoning", "tools", "structured_output"},
            ),
            _profile("gpt-5.6-luna", "1.5", "12", "0.18", {"text"}),
            _profile("gpt-5.6-sol", "5", "30", "0.5", {"text"}),
            _profile("gpt-5.6-terra", "2.5", "20", "0.3", {"text"}),
        )
    )


def default_routes(registry: ModelRegistry | None = None) -> dict[str, RouteProfile]:
    selected = registry or default_model_registry()
    definitions = {
        "explain": frozenset({"text"}),
        "change-plan": frozenset({"text", "tools", "structured_output"}),
        "patch": frozenset({"text", "tools", "structured_output"}),
        "patch-review": frozenset({"text", "structured_output"}),
    }
    routes: dict[str, RouteProfile] = {}
    for scene, required in definitions.items():
        routes[scene] = RouteProfile(
            scene,
            DEFAULT_MODEL_ID,
            selected.cheaper_capable_fallbacks(DEFAULT_MODEL_ID, required),
            required,
        )
    return routes
