from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from portfolio.models import TradeCandidate


def _dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def candidate_to_dict(candidate: TradeCandidate) -> dict[str, Any]:
    return {
        "schema_version": candidate.schema_version,
        "candidate_id": candidate.candidate_id,
        "run_id": str(candidate.run_id),
        "market_id": candidate.market_id,
        "event_id": candidate.event_id,
        "symbol": candidate.symbol,
        "asset_name": candidate.asset_name,
        "pass_number": candidate.pass_number,
        "event_archetype": candidate.event_archetype,
        "strategy_branch": candidate.strategy_branch,
        "portfolio_label": candidate.portfolio_label,
        "direction": candidate.direction,
        "consumes_capital": candidate.consumes_capital,
        "trigger_at": candidate.trigger_at.isoformat(),
        "entry_at": candidate.entry_at.isoformat(),
        "resolution": candidate.resolution,
        "entry_price": candidate.entry_price,
        "exit_at": candidate.exit_at.isoformat(),
        "exit_price": candidate.exit_price,
        "exit_reason": candidate.exit_reason,
        "path_reference_quantity": candidate.path_reference_quantity,
        "initial_stop": candidate.initial_stop,
        "atr": candidate.atr,
        "stop_distance": candidate.stop_distance,
        "range_period": candidate.range_period,
        "range_multiplier": candidate.range_multiplier,
        "classification_probability": candidate.classification_probability,
        "predicted_peak_percent": candidate.predicted_peak_percent,
        "predicted_target_price": candidate.predicted_target_price,
        "remaining_gap": candidate.remaining_gap,
        "directions_agree": candidate.directions_agree,
        "price_momentum": candidate.price_momentum,
        "probability_at_entry": candidate.probability_at_entry,
        "sector_key": candidate.sector_key,
        "theme_tags": list(candidate.theme_tags),
        "country_tags": list(candidate.country_tags),
        "adv": candidate.adv,
        "polymarket_volume_quality": candidate.polymarket_volume_quality,
        "question": candidate.question,
        "final_outcome": candidate.final_outcome,
        "bar_lookup_key": candidate.bar_lookup_key,
        "bar_window": {
            "start": candidate.bar_window_start.isoformat(),
            "end": candidate.bar_window_end.isoformat(),
        },
        "momentum_parameter_selection": candidate.momentum_parameter_selection,
    }


def candidate_from_dict(data: dict[str, Any]) -> TradeCandidate:
    bar_window = data.get("bar_window") or {}
    return TradeCandidate(
        schema_version=int(data.get("schema_version", 1)),
        candidate_id=str(data["candidate_id"]),
        run_id=UUID(str(data["run_id"])),
        market_id=str(data["market_id"]),
        event_id=str(data["event_id"]),
        symbol=str(data["symbol"]),
        asset_name=str(data.get("asset_name", "")),
        pass_number=int(data["pass_number"]),
        event_archetype=data.get("event_archetype"),
        strategy_branch=str(data["strategy_branch"]),
        portfolio_label=str(data["portfolio_label"]),
        direction=data["direction"],
        consumes_capital=bool(data["consumes_capital"]),
        trigger_at=_dt(data["trigger_at"]),
        entry_at=_dt(data["entry_at"]),
        resolution=data["resolution"],
        entry_price=float(data["entry_price"]),
        exit_at=_dt(data["exit_at"]),
        exit_price=float(data["exit_price"]),
        exit_reason=str(data["exit_reason"]),
        path_reference_quantity=float(data["path_reference_quantity"]),
        initial_stop=float(data["initial_stop"]),
        atr=float(data["atr"]),
        stop_distance=float(data["stop_distance"]),
        range_period=int(data["range_period"]),
        range_multiplier=float(data["range_multiplier"]),
        classification_probability=data.get("classification_probability"),
        predicted_peak_percent=data.get("predicted_peak_percent"),
        predicted_target_price=data.get("predicted_target_price"),
        remaining_gap=data.get("remaining_gap"),
        directions_agree=data.get("directions_agree"),
        price_momentum=data.get("price_momentum"),
        probability_at_entry=data.get("probability_at_entry"),
        sector_key=str(data.get("sector_key", "UNKNOWN")),
        theme_tags=tuple(data.get("theme_tags") or ()),
        country_tags=tuple(data.get("country_tags") or ()),
        adv=data.get("adv"),
        polymarket_volume_quality=dict(data.get("polymarket_volume_quality") or {}),
        question=str(data.get("question", "")),
        final_outcome=data.get("final_outcome"),
        bar_lookup_key=str(data["bar_lookup_key"]),
        bar_window_start=_dt(bar_window.get("start", data["entry_at"])),
        bar_window_end=_dt(bar_window.get("end", data["exit_at"])),
        momentum_parameter_selection=data.get("momentum_parameter_selection"),
    )
