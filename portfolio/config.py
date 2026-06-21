from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

DEFAULT_GEO_COUNTRY_TAGS = frozenset(
    {
        "china",
        "iran",
        "israel",
        "middle-east",
        "russia",
        "strait-of-hormuz",
        "ukraine",
    }
)


@dataclass(frozen=True)
class PortfolioConfig:
    starting_capital: float = 100_000.0
    sizing_mode: Literal["risk_based", "fixed_fraction"] = "risk_based"
    risk_per_trade_pct: float = 0.01
    fixed_fraction_pct: float = 0.02
    max_position_notional_pct: float = 0.10
    max_open_positions: int = 20
    one_position_per_event_symbol: bool = True
    min_position_notional: float = 100.0
    max_event_exposure_pct: float = 0.15
    max_sector_exposure_pct: float = 0.25
    max_theme_exposure_pct: float = 0.30
    max_country_exposure_pct: float = 0.30
    max_gross_exposure_pct: float = 1.50
    max_net_exposure_pct: float = 1.00
    max_portfolio_heat_pct: float = 0.06
    max_adv_participation_pct: float = 0.05
    adv_lookback_days: int = 20
    adv_min_bars: int = 5
    drawdown_derisk_schedule: dict[float, float] = field(
        default_factory=lambda: {0.10: 0.5, 0.20: 0.0}
    )
    kill_switch_drawdown_pct: float = 0.20
    allow_shorts: bool = True
    short_margin_pct: float = 1.00
    confidence_weighting_enabled: bool = False
    paper_trade_enabled: bool = False
    on_missing_liquidity: Literal["skip_warn", "paper_trade", "reject"] = "skip_warn"
    on_missing_volume: Literal["skip_warn", "paper_trade", "reject"] = "skip_warn"
    short_route: Literal["reject", "paper"] = "reject"
    slippage_bps: float = 0.0
    geo_country_tags: frozenset[str] = DEFAULT_GEO_COUNTRY_TAGS
    theme_tags: frozenset[str] | None = None
    caps_enabled: bool = True

    def with_passthrough_profile(self) -> PortfolioConfig:
        return replace(
            self,
            starting_capital=1e12,
            risk_per_trade_pct=1.0,
            max_position_notional_pct=1.0,
            max_event_exposure_pct=1.0,
            max_sector_exposure_pct=1.0,
            max_theme_exposure_pct=1.0,
            max_country_exposure_pct=1.0,
            max_gross_exposure_pct=1.0,
            max_net_exposure_pct=1.0,
            max_portfolio_heat_pct=1.0,
            max_open_positions=999_999,
            min_position_notional=0.0,
            one_position_per_event_symbol=False,
            kill_switch_drawdown_pct=1.0,
            drawdown_derisk_schedule={},
            caps_enabled=False,
        )

    def with_caps_disabled(self) -> PortfolioConfig:
        return self.with_passthrough_profile()

    def resolved_theme_tags(self, included_tags: frozenset[str]) -> frozenset[str]:
        if self.theme_tags is not None:
            return self.theme_tags
        return included_tags - self.geo_country_tags

    def to_json(self) -> dict[str, Any]:
        values = asdict(self)
        values["geo_country_tags"] = sorted(self.geo_country_tags)
        if self.theme_tags is not None:
            values["theme_tags"] = sorted(self.theme_tags)
        values["drawdown_derisk_schedule"] = {
            str(key): value for key, value in self.drawdown_derisk_schedule.items()
        }
        return values

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> PortfolioConfig:
        parsed = dict(values)
        parsed["geo_country_tags"] = frozenset(parsed.get("geo_country_tags", DEFAULT_GEO_COUNTRY_TAGS))
        theme_tags = parsed.get("theme_tags")
        parsed["theme_tags"] = frozenset(theme_tags) if theme_tags is not None else None
        schedule = parsed.get("drawdown_derisk_schedule", {0.10: 0.5, 0.20: 0.0})
        parsed["drawdown_derisk_schedule"] = {
            float(key): float(value) for key, value in schedule.items()
        }
        parsed.setdefault("caps_enabled", True)
        parsed.setdefault("adv_lookback_days", 20)
        parsed.setdefault("adv_min_bars", 5)
        return cls(**parsed)
