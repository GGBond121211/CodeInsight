"""截图价格与能力约束对应的版本化模型注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

PRICE_VERSION = "aliyun-maas-2026-09-11"
# 2026-09-11：旧中转站额度耗尽，换到与 Embedding 同一家的阿里云 MaaS 兼容端点。
# 价格来自控制台模型广场截图（元/百万 token）：qwen3.8-flash 输入 0.8、输出 2.7；
# qwen3.7-flash 输入 0.2、输出 0.8。
#
# cached_input_price_per_million 没有查到中转站的缓存计价：这里取与输入价相同的值，
# 含义是「按不命中缓存计算」，成本估计因此偏保守，不会把未证实的折扣写进账。
DEFAULT_MODEL_ID = "qwen3.8-flash"
FALLBACK_MODEL_ID = "qwen3.7-flash-2026-07-15"


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
    """登记可用模型；未明确能力的模型不进入自动降级链。

    2026-09-11 起默认模型是阿里云 MaaS 的 ``qwen3.8-flash``，自动降级链只有一个候选：
    ``qwen3.7-flash-2026-07-15``（价格更低、能力相同）。旧中转站的模型仍留在目录里作为
    历史价格与对照，但既不是默认模型，也不带 ``fallback_eligible``。
    """
    return ModelRegistry(
        (
            _profile("codex-auto-review", "1.75", "14", "0.175", {"text"}),
            _profile(
                DEFAULT_MODEL_ID,
                "0.8",
                "2.7",
                "0.8",
                {"text", "tools", "structured_output"},
                quality_tier="best",
            ),
            _profile(
                FALLBACK_MODEL_ID,
                "0.2",
                "0.8",
                "0.2",
                {"text", "tools", "structured_output"},
                fallback_eligible=True,
            ),
            # 2026-09-11：不再是默认模型。新中转站上它的免费额度已耗尽，
            # 报价也比 qwen3.8-flash 高出十倍以上，因此不再带 best 标记与降级资格。
            _profile(
                "deepseek-v4-flash",
                "12",
                "36",
                "0.3996",
                {"text", "tools", "structured_output"},
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
        "general-chat": frozenset({"text"}),
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
