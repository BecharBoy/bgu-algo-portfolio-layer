from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from database.backtesting.market_data import bars_before
from database.backtesting.repositories import json_value
from database.backtesting.repositories.prices import asset_metadata, price_bars
from database.backtesting.repositories.trades import save_trade
from main_backtesting.models import PriceBar, SourceEvent, SourceMarket, Trade
from portfolio.candidate_builder import (
    atr_from_trade,
    build_trade_candidate,
    compute_adv,
    infer_archetype,
    partition_tags,
)
from portfolio.config import PortfolioConfig
from portfolio.models import TradeCandidate
from portfolio.mtm import build_bar_index
from portfolio.portfolio import Portfolio
from portfolio.replay import replay_portfolio
from portfolio.reporting import generate_portfolio_reports
from portfolio.serialization import candidate_from_dict, candidate_to_dict
from strategies.event_driven_long import average_true_range


async def load_pass1_candidates_from_work(
    conn: Any,
    *,
    run_id: object,
) -> list[TradeCandidate]:
    rows = await conn.fetch(
        """
        SELECT result
        FROM checking_relevant_events.historical_backtest_stage_work
        WHERE run_id = $1 AND stage = 'simulation' AND status = 'complete'
        """,
        run_id,
    )
    candidates: list[TradeCandidate] = []
    for row in rows:
        result = json_value(row["result"]) or {}
        for item in result.get("pass1_candidates") or []:
            candidates.append(candidate_from_dict(item))
    return candidates


async def assemble_candidate(
    engine: Any,
    conn: Any,
    *,
    trade: Trade,
    market: SourceMarket,
    event: SourceEvent,
    bars: list[PriceBar],
    daily_bars: list[PriceBar],
    bar_window_start: datetime,
    bar_window_end: datetime,
    polymarket_volume_quality: dict[str, Any],
    consumes_capital: bool,
    observation_archetype: str | None,
    classification_probability: float | None = None,
    predicted_peak_percent: float | None = None,
    remaining_gap: float | None = None,
    directions_agree: bool | None = None,
    price_momentum: float | None = None,
    probability_at_entry: float | None = None,
) -> TradeCandidate | None:
    metadata = await asset_metadata(conn, trade.symbol)
    sector_key = "UNKNOWN"
    if metadata and metadata["benchmark_symbol"]:
        sector_key = str(metadata["benchmark_symbol"])
    portfolio_config: PortfolioConfig = engine.config.portfolio_config
    theme_tags, country_tags = partition_tags(
        list(event.tags),
        list(market.tags),
        theme_tags=set(portfolio_config.resolved_theme_tags(engine.config.included_tags)),
        geo_country_tags=set(portfolio_config.geo_country_tags),
    )
    previous = bars_before(bars, trade.entry_at)
    period = trade.range_period or engine.config.trailing_range_bars
    atr = atr_from_trade(trade, previous, period) or average_true_range(previous, period)
    if atr is None:
        atr = abs(trade.entry_price - trade.initial_stop)
    adv = compute_adv(
        daily_bars,
        trade.entry_at,
        lookback=portfolio_config.adv_lookback_days,
        min_bars=portfolio_config.adv_min_bars,
    )
    return build_trade_candidate(
        trade=trade,
        atr=float(atr),
        consumes_capital=consumes_capital,
        sector_key=sector_key,
        theme_tags=theme_tags,
        country_tags=country_tags,
        adv=adv,
        polymarket_volume_quality=polymarket_volume_quality,
        bar_window_start=bar_window_start,
        bar_window_end=bar_window_end,
        event_archetype_value=infer_archetype(
            observation_archetype=observation_archetype,
            event_tags=list(event.tags),
            question=market.question,
            symbol=trade.symbol,
        ),
        classification_probability=classification_probability,
        predicted_peak_percent=predicted_peak_percent,
        remaining_gap=remaining_gap,
        directions_agree=directions_agree,
        price_momentum=price_momentum,
        probability_at_entry=probability_at_entry,
    )


async def load_bars_for_candidates(conn: Any, candidates: list[TradeCandidate]) -> dict[str, list[PriceBar]]:
    bars_by_key: dict[str, list[PriceBar]] = {}
    for candidate in candidates:
        key = candidate.bar_lookup_key
        if key in bars_by_key:
            continue
        symbol, resolution = key.split(":", 1)
        bars_by_key[key] = await price_bars(
            conn,
            symbol=symbol,
            resolution=resolution,  # type: ignore[arg-type]
            start=candidate.bar_window_start,
            end=candidate.bar_window_end,
        )
    return bars_by_key


async def run_portfolio_pass2(
    engine: Any,
    conn: Any,
    candidates: list[TradeCandidate],
) -> dict[str, Any]:
    if not candidates:
        return {"booked_trades": 0, "decisions": 0}
    bars_by_key = await load_bars_for_candidates(conn, candidates)
    portfolio = Portfolio.empty(engine.config.portfolio_config)
    portfolio.set_bars(build_bar_index(bars_by_key))
    replay_portfolio(portfolio, candidates)
    await conn.execute(
        """
        DELETE FROM checking_relevant_events.historical_trades
        WHERE run_id = $1
          AND portfolio IN ('machine_learning', 'polymarket_momentum')
        """,
        engine.run_id,
    )
    for trade in portfolio.booked_trades:
        await save_trade(conn, trade)
    report_dir = engine.run_dir / "reports"
    generate_portfolio_reports(portfolio, report_dir)
    return {
        "booked_trades": len(portfolio.booked_trades),
        "decisions": len(portfolio.decisions),
        "final_equity": portfolio.state.equity,
    }
