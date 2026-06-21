from __future__ import annotations

import math
from dataclasses import dataclass

from portfolio.config import PortfolioConfig
from portfolio.models import CAP_ORDER, PortfolioState, TradeCandidate
from strategies.event_driven_long import ib_style_commission


@dataclass(frozen=True)
class CapLimit:
    name: str
    max_notional: float


@dataclass(frozen=True)
class NormalizedCandidate:
    candidate: TradeCandidate
    active_caps: tuple[str, ...]
    warnings: tuple[str, ...]
    skip_theme_cap: bool
    skip_country_cap: bool
    skip_liquidity_cap: bool
    skip_volume_gate: bool


def normalize_candidate(
    candidate: TradeCandidate,
    config: PortfolioConfig,
) -> tuple[NormalizedCandidate | None, str | None]:
    warnings: list[str] = []
    active: list[str] = [
        "max_position_size",
        "max_event_exposure",
        "max_sector_exposure",
        "max_gross_exposure",
        "max_portfolio_heat",
        "insufficient_cash",
    ]
    if candidate.entry_price <= 0 or math.isnan(candidate.entry_price):
        return None, "invalid_entry_price"
    if candidate.stop_distance <= 0 or math.isnan(candidate.stop_distance):
        return None, "invalid_stop_distance"
    if not candidate.symbol or not candidate.event_id or candidate.direction not in {"long", "short"}:
        return None, "invalid_candidate"

    skip_theme = not candidate.theme_tags
    skip_country = not candidate.country_tags
    skip_liquidity = candidate.adv is None or candidate.adv <= 0
    skip_volume = not candidate.polymarket_volume_quality or candidate.polymarket_volume_quality.get(
        "reason"
    ) == "polymarket_volume_unavailable"

    if skip_theme:
        warnings.append("missing_theme_tags")
    else:
        active.append("max_theme_exposure")
    if skip_country:
        pass
    else:
        active.append("max_country_exposure")
    if skip_liquidity:
        warnings.append("missing_adv")
    else:
        active.append("liquidity_adv")
    if skip_volume:
        warnings.append("missing_polymarket_volume_quality")

    if candidate.sector_key == "UNKNOWN":
        warnings.append("missing_sector_benchmark")

    return (
        NormalizedCandidate(
            candidate=candidate,
            active_caps=tuple(active),
            warnings=tuple(warnings),
            skip_theme_cap=skip_theme,
            skip_country_cap=skip_country,
            skip_liquidity_cap=skip_liquidity,
            skip_volume_gate=skip_volume,
        ),
        None,
    )


def volume_gate_reject(normalized: NormalizedCandidate) -> str | None:
    quality = normalized.candidate.polymarket_volume_quality
    if normalized.skip_volume_gate:
        return None
    if quality.get("allowed") is False:
        return "low_polymarket_volume_quality"
    return None


def compute_caps(
    *,
    config: PortfolioConfig,
    state: PortfolioState,
    normalized: NormalizedCandidate,
    equity: float,
    entry_price: float,
    stop_distance: float,
) -> list[CapLimit]:
    if not config.caps_enabled:
        return [CapLimit("insufficient_cash", equity * 1e6)]
    candidate = normalized.candidate
    caps = [
        CapLimit("max_position_size", config.max_position_notional_pct * equity),
        CapLimit(
            "max_event_exposure",
            config.max_event_exposure_pct * equity - state.event_exposure.get(candidate.event_id, 0.0),
        ),
        CapLimit(
            "max_sector_exposure",
            config.max_sector_exposure_pct * equity
            - state.sector_exposure.get(candidate.sector_key, 0.0),
        ),
        CapLimit(
            "max_gross_exposure",
            config.max_gross_exposure_pct * equity - state.gross_exposure,
        ),
        CapLimit(
            "max_net_exposure",
            _net_headroom(config, state, equity, candidate.direction, entry_price, stop_distance),
        ),
        CapLimit(
            "max_portfolio_heat",
            ((config.max_portfolio_heat_pct * equity - state.heat) * entry_price / stop_distance)
            if stop_distance > 0
            else 0.0,
        ),
        CapLimit("insufficient_cash", max(state.cash, 0.0)),
    ]
    if not normalized.skip_theme_cap:
        for tag in candidate.theme_tags:
            caps.append(
                CapLimit(
                    "max_theme_exposure",
                    config.max_theme_exposure_pct * equity - state.theme_exposure.get(tag, 0.0),
                )
            )
    if not normalized.skip_country_cap:
        for tag in candidate.country_tags:
            caps.append(
                CapLimit(
                    "max_country_exposure",
                    config.max_country_exposure_pct * equity - state.country_exposure.get(tag, 0.0),
                )
            )
    if not normalized.skip_liquidity_cap and candidate.adv is not None:
        caps.append(
            CapLimit(
                "liquidity_adv",
                config.max_adv_participation_pct * candidate.adv * entry_price,
            )
        )
    return caps


def apply_caps(
    *,
    notional_risk: float,
    caps: list[CapLimit],
    min_notional: float,
) -> tuple[float, str | None]:
    binding: str | None = None
    final_notional = notional_risk
    ordered: list[CapLimit] = []
    for name in CAP_ORDER:
        ordered.extend([cap for cap in caps if cap.name == name])
    for cap in ordered:
        if cap.max_notional < final_notional:
            final_notional = max(cap.max_notional, 0.0)
            binding = cap.name
        elif cap.max_notional == final_notional and binding is None:
            binding = cap.name
    if final_notional < min_notional:
        return final_notional, binding or "below_min_notional"
    return final_notional, binding


def estimate_commission(quantity: float, notional: float) -> float:
    try:
        return ib_style_commission(quantity, notional)
    except Exception:
        return max(1.0, 0.01 * notional)


def _net_headroom(
    config: PortfolioConfig,
    state: PortfolioState,
    equity: float,
    direction: str,
    entry_price: float,
    stop_distance: float,
) -> float:
    del entry_price, stop_distance
    max_net = config.max_net_exposure_pct * equity
    if direction == "long":
        return max(max_net - state.net_exposure, 0.0)
    return max(max_net + state.net_exposure, 0.0)
