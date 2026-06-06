from __future__ import annotations

import asyncio
import shutil
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from database.backtesting.market_data import (
    YFinanceHourlyClient,
    bars_before,
    bars_from,
    next_bar_after,
)
from database.backtesting.news import GdeltNewsClient
from database.backtesting.polymarket import PolymarketHistoryClient, probability_as_of
from database.backtesting.repository import (
    candidate_events,
    create_run,
    event_markets,
    finish_run,
    save_articles,
    save_asset_world,
    save_event_decision,
    save_passes,
    save_sentiment,
    save_skip,
    save_trade,
)
from database.backtesting.schema import reset_backtesting_schema
from database.backtesting.sentiment import FinbertSentimentAnalyzer
from database.db_connection import connect
from LLM.build_world import assets_from_world, build_asset_world
from LLM.news_sentiment_shadow import analyze_with_ollama
from LLM.ollama_client import OllamaClient
from LLM.remove_unwanted_markets import classify_event
from main_backtesting.config import BacktestConfig
from main_backtesting.models import ProbabilityPoint, SourceMarket, Trade
from main_backtesting.reporting import (
    create_trade_graph,
    export_database_reports,
    write_reports,
)
from strategies.event_driven_long import EventDrivenLongStrategy, ThresholdPassTracker


def _config_json(config: BacktestConfig) -> dict[str, Any]:
    value = asdict(config)
    for key, item in value.items():
        if isinstance(item, Path):
            value[key] = str(item)
        elif isinstance(item, frozenset):
            value[key] = sorted(item)
        elif hasattr(item, "isoformat"):
            value[key] = item.isoformat()
    return value


def clear_output_directories(config: BacktestConfig) -> None:
    resolved_output = config.output_dir.resolve()
    resolved_root = (config.output_dir.parent).resolve()
    if resolved_output.parent != resolved_root:
        raise RuntimeError(f"Refusing to clear unexpected output path: {resolved_output}")
    if resolved_output.exists():
        shutil.rmtree(resolved_output)
    config.graph_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)


def detect_passes(
    market_id: str,
    probabilities: list[ProbabilityPoint],
    threshold: float,
) -> list:
    tracker = ThresholdPassTracker(market_id, threshold)
    for point in probabilities:
        tracker.observe(point.timestamp, point.probability)
    return tracker.passes


class BacktestEngine:
    def __init__(self, config: BacktestConfig, *, max_events: int = 0) -> None:
        self.config = config
        self.max_events = max_events
        self.run_id: UUID = uuid4()
        self.ollama = OllamaClient()
        self.polymarket = PolymarketHistoryClient(
            chunk_days=self.config.probability_chunk_days
        )
        self.prices = YFinanceHourlyClient()
        self.news = GdeltNewsClient(max_articles=self.config.max_articles)
        self.finbert = FinbertSentimentAnalyzer()
        self.strategy = EventDrivenLongStrategy(
            trade_notional=self.config.trade_notional,
            range_period=self.config.trailing_range_bars,
            range_multiplier=self.config.trailing_range_multiplier,
        )
        self.trades: list[Trade] = []

    async def close(self) -> None:
        await asyncio.gather(
            self.ollama.close(),
            self.polymarket.close(),
            self.news.close(),
        )

    async def run(self) -> UUID:
        clear_output_directories(self.config)
        conn = await connect()
        try:
            await reset_backtesting_schema(conn)
            await create_run(conn, self.run_id, _config_json(self.config))
            events = await candidate_events(
                conn,
                start=self.config.start,
                end=self.config.end,
                minimum_days_remaining=self.config.minimum_days_remaining,
                maximum_days_remaining=self.config.maximum_days_remaining,
                included_tags=sorted(self.config.included_tags),
                excluded_tags=sorted(self.config.excluded_tags),
                limit=self.max_events,
            )
            print(f"[events] candidates={len(events)}")
            for event_index, event in enumerate(events, start=1):
                try:
                    decision = await classify_event(self.ollama, event)
                except Exception as exc:
                    await save_skip(
                        conn,
                        self.run_id,
                        event_id=event.event_id,
                        stage="event_filter",
                        reason=str(exc),
                    )
                    continue
                await save_event_decision(
                    conn,
                    self.run_id,
                    event,
                    decision.relevant_to_financial_markets,
                    decision.reason,
                    decision.model_dump(mode="json"),
                )
                print(
                    f"[event {event_index}/{len(events)}] relevant="
                    f"{decision.relevant_to_financial_markets} {event.title}"
                )
                if not decision.relevant_to_financial_markets:
                    continue
                for market in await event_markets(conn, event):
                    try:
                        await self._process_market(conn, market)
                    except Exception as exc:
                        await save_skip(
                            conn,
                            self.run_id,
                            event_id=market.event_id,
                            market_id=market.market_id,
                            stage="market_processing",
                            reason=str(exc),
                        )

            write_reports(self.trades, self.config.report_dir)
            await export_database_reports(conn, self.run_id, self.config.report_dir)
            await finish_run(conn, self.run_id, status="complete")
            return self.run_id
        except Exception as exc:
            await finish_run(conn, self.run_id, status="failed", error=str(exc))
            raise
        finally:
            await conn.close()
            await self.close()

    async def _process_market(self, conn: Any, market: SourceMarket) -> None:
        probabilities = await self.polymarket.hourly_probabilities(
            market,
            start=self.config.start,
            end=self.config.end,
        )
        passes = detect_passes(market.market_id, probabilities, self.config.threshold)
        await save_passes(conn, self.run_id, market, passes)
        if not passes:
            return

        world = await build_asset_world(self.ollama, market)
        await save_asset_world(
            conn,
            self.run_id,
            market.market_id,
            self.ollama.model_name,
            world.model_dump(mode="json"),
        )
        assets = assets_from_world(world)
        print(f"  [market] passes={len(passes)} assets={len(assets)} {market.question}")

        for threshold_pass in passes:
            for asset in assets:
                try:
                    await self._consider_trade(
                        conn,
                        market=market,
                        probabilities=probabilities,
                        pass_number=threshold_pass.pass_number,
                        trigger_at=threshold_pass.above_at,
                        asset=asset,
                    )
                except Exception as exc:
                    await save_skip(
                        conn,
                        self.run_id,
                        event_id=market.event_id,
                        market_id=market.market_id,
                        pass_number=threshold_pass.pass_number,
                        symbol=asset.symbol,
                        stage="asset_processing",
                        reason=str(exc),
                    )

    async def _consider_trade(
        self,
        conn: Any,
        *,
        market: SourceMarket,
        probabilities: list[ProbabilityPoint],
        pass_number: int,
        trigger_at: Any,
        asset: Any,
    ) -> None:
        news_start = trigger_at - self.config.news_lookback
        articles = await self.news.articles(
            market=market,
            asset=asset,
            start=news_start,
            end=trigger_at,
        )
        await save_articles(
            conn,
            self.run_id,
            market.market_id,
            pass_number,
            asset.symbol,
            articles,
        )
        finbert = await self.finbert.analyze(articles)
        shadow = await analyze_with_ollama(
            self.ollama,
            market=market,
            asset=asset,
            articles=articles,
        )
        await save_sentiment(
            conn,
            self.run_id,
            market.market_id,
            pass_number,
            asset.symbol,
            "finbert",
            finbert,
        )
        await save_sentiment(
            conn,
            self.run_id,
            market.market_id,
            pass_number,
            asset.symbol,
            "ollama_shadow",
            shadow,
        )
        if finbert.label != "positive":
            await save_skip(
                conn,
                self.run_id,
                event_id=market.event_id,
                market_id=market.market_id,
                pass_number=pass_number,
                symbol=asset.symbol,
                stage="finbert_gate",
                reason=f"FinBERT label={finbert.label}",
            )
            return

        price_start = self.config.start - timedelta(days=7)
        bars = await self.prices.hourly_bars(
            asset.symbol,
            start=price_start,
            end=self.config.end,
        )
        entry_bar = next_bar_after(bars, trigger_at)
        if entry_bar is None or entry_bar.timestamp >= self.config.end:
            await save_skip(
                conn,
                self.run_id,
                event_id=market.event_id,
                market_id=market.market_id,
                pass_number=pass_number,
                symbol=asset.symbol,
                stage="entry_price",
                reason="No yfinance hourly entry bar before simulation end",
            )
            return

        latest_probability = probability_as_of(probabilities, entry_bar.timestamp)
        if latest_probability is None or latest_probability <= self.config.threshold:
            await save_skip(
                conn,
                self.run_id,
                event_id=market.event_id,
                market_id=market.market_id,
                pass_number=pass_number,
                symbol=asset.symbol,
                stage="probability_recheck",
                reason=f"Probability at entry was {latest_probability}",
            )
            return

        previous_bars = bars_before(bars, entry_bar.timestamp)
        trade = self.strategy.open_trade(
            run_id=self.run_id,
            market_id=market.market_id,
            event_id=market.event_id,
            question=market.question,
            symbol=asset.symbol,
            asset_name=asset.asset_name,
            pass_number=pass_number,
            trigger_at=trigger_at,
            entry_bar=entry_bar,
            previous_bars=previous_bars,
            final_outcome=market.final_outcome,
        )
        if trade is None:
            await save_skip(
                conn,
                self.run_id,
                event_id=market.event_id,
                market_id=market.market_id,
                pass_number=pass_number,
                symbol=asset.symbol,
                stage="trailing_stop_setup",
                reason="Insufficient completed yfinance hourly bars for initial trailing stop",
            )
            return

        for bar in bars_from(bars, entry_bar.timestamp):
            prior = bars_before(bars, bar.timestamp)
            if self.strategy.update_trade(trade, bar, prior):
                break
        if trade.exit_at is None:
            final_bars = [bar for bar in bars if bar.timestamp < self.config.end]
            if final_bars:
                trade.final_mark_price = final_bars[-1].close

        graph = create_trade_graph(
            trade,
            bars=bars,
            probabilities=probabilities,
            simulation_end=self.config.end,
            graph_dir=self.config.graph_dir,
        )
        trade.graph_path = str(graph)
        await save_trade(conn, trade)
        self.trades.append(trade)
        print(
            f"    [trade] pass={pass_number} {asset.symbol} "
            f"net={trade.net_profit if trade.net_profit is not None else 'n/a'}"
        )
