from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID
from database.backtesting.repositories.research import (
    link_run_sentiment,
    reusable_article_set,
    reusable_sentiment,
    save_article_set,
    save_sentiment,
)
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.worlds import run_world_assets
from LLM.news_sentiment_shadow import BATCH_SYSTEM_PROMPT as SENTIMENT_BATCH_PROMPT, analyze_batch_with_ollama, batch_request_payload
from main_backtesting.models import Asset, NewsArticle, SourceMarket
from main_backtesting.utils import chunks, input_hash

from main_backtesting.stages.event_filter import accepted_markets


async def run(self, conn: Any) -> None:
    market_map = {market.market_id: market for market in await accepted_markets(self, conn)}
    jobs = list(await run_world_assets(conn, self.run_id))
    for batch_index, batch in enumerate(chunks(jobs, self.config.gdelt_concurrency), 1):
        active: list[tuple[Any, SourceMarket, Asset, str, datetime, UUID | None, list[NewsArticle] | None]] = []
        for row in batch:
            work_key = f"{row['market_id']}:{row['pass_number']}:{row['symbol']}"
            self.current_work_key = work_key
            if not await start_work(
                conn,
                run_id=self.run_id,
                stage="research",
                work_key=work_key,
                payload={"market_id": row["market_id"], "pass_number": row["pass_number"], "symbol": row["symbol"]},
            ):
                continue
            market = market_map[row["market_id"]]
            asset = Asset(row["symbol"], row["asset_name"], row["asset_class"], row["reason"])
            start = row["as_of"] - self.config.news_lookback
            query = self.news.query(market, asset)
            article_identity = input_hash(
                {
                    "task": "gdelt_articles",
                    "market_id": market.market_id,
                    "pass_number": row["pass_number"],
                    "as_of": row["as_of"],
                    "symbol": asset.symbol,
                    "query": query,
                    "start": start,
                    "end": row["as_of"],
                    "max_articles": self.config.max_articles,
                }
            )
            cached = await reusable_article_set(conn, article_identity)
            active.append(
                (
                    row,
                    market,
                    asset,
                    article_identity,
                    start,
                    cached[0] if cached else None,
                    cached[1] if cached else None,
                )
            )
        downloads = await asyncio.gather(
            *[
                self.news.articles(
                    market=item[1], asset=item[2], start=item[4], end=item[0]["as_of"]
                )
                for item in active
                if item[6] is None
            ]
        )
        download_index = 0
        prepared: list[tuple[Any, SourceMarket, Asset, UUID, list[NewsArticle]]] = []
        for row, market, asset, article_identity, start, article_set_id, articles in active:
            if articles is None:
                articles = downloads[download_index]
                download_index += 1
                article_set_id = await save_article_set(
                    conn,
                    input_hash=article_identity,
                    market_id=market.market_id,
                    pass_number=row["pass_number"],
                    as_of=row["as_of"],
                    symbol=asset.symbol,
                    query=self.news.query(market, asset),
                    window_start=start,
                    window_end=row["as_of"],
                    query_settings={"max_articles": self.config.max_articles, "source": "GDELT"},
                    articles=articles,
                )
            prepared.append((row, market, asset, article_set_id, articles))

        ollama_requests: list[tuple[str, SourceMarket, Asset, list[NewsArticle]]] = []
        prepared_by_id: dict[str, tuple[Any, SourceMarket, Asset, UUID]] = {}
        for row, market, asset, article_set_id, articles in prepared:
            finbert_identity = input_hash(
                {"task": "sentiment", "provider": "finbert", "article_set_id": article_set_id}
            )
            finbert = await reusable_sentiment(conn, finbert_identity)
            if finbert is None:
                finbert = await self.finbert.analyze(articles)
                await save_sentiment(
                    conn,
                    run_id=self.run_id,
                    input_hash=finbert_identity,
                    article_set_id=article_set_id,
                    market_id=market.market_id,
                    pass_number=row["pass_number"],
                    symbol=asset.symbol,
                    provider="finbert",
                    model_name="ProsusAI/finbert",
                    prompt_version="finbert-default",
                    model_input={
                        "article_set_id": str(article_set_id),
                        "article_urls": [article.url for article in articles],
                    },
                    model_output={"details": finbert.details},
                    result=finbert,
                )
            else:
                await link_run_sentiment(
                    conn,
                    run_id=self.run_id,
                    market_id=market.market_id,
                    pass_number=row["pass_number"],
                    symbol=asset.symbol,
                    provider="finbert",
                    input_hash=finbert_identity,
                )
            request_id = f"{market.market_id}:{row['pass_number']}:{asset.symbol}"
            ollama_requests.append((request_id, market, asset, articles))
            prepared_by_id[request_id] = (row, market, asset, article_set_id)
        for sentiment_batch in chunks(ollama_requests, self.config.ollama_sentiment_batch_size):
            batch_model_input = {
                "system_prompt": SENTIMENT_BATCH_PROMPT,
                "payload": batch_request_payload(sentiment_batch),
            }
            identities = {
                request_id: input_hash(
                    {
                        "task": "sentiment",
                        "provider": "ollama",
                        "article_set_id": str(prepared_by_id[request_id][3]),
                        "model": self.ollama.model_name,
                        "prompt_version": self.config.ollama_sentiment_prompt_version,
                        "model_input": batch_model_input,
                        "request_id": request_id,
                    }
                )
                for request_id, _, _, _ in sentiment_batch
            }
            cached = {
                request_id: await reusable_sentiment(conn, identities[request_id])
                for request_id, _, _, _ in sentiment_batch
            }
            cached_count = sum(item is not None for item in cached.values())
            if cached_count not in {0, len(sentiment_batch)}:
                raise RuntimeError(
                    f"Partial exact Ollama-sentiment batch cache in research batch {batch_index}"
                )
            if cached_count == len(sentiment_batch):
                async with conn.transaction():
                    for request_id, _, asset, _ in sentiment_batch:
                        row, market, _, _ = prepared_by_id[request_id]
                        await link_run_sentiment(
                            conn,
                            run_id=self.run_id,
                            market_id=market.market_id,
                            pass_number=row["pass_number"],
                            symbol=asset.symbol,
                            provider="ollama",
                            input_hash=identities[request_id],
                        )
                continue
            results = await analyze_batch_with_ollama(self.ollama, sentiment_batch)
            batch_model_output = {
                "sentiments": [
                    {
                        "request_id": request_id,
                        "label": result.label,
                        "score": result.score,
                        "details": result.details,
                    }
                    for request_id, result in results.items()
                ]
            }
            async with conn.transaction():
                for request_id, result in results.items():
                    row, market, asset, article_set_id = prepared_by_id[request_id]
                    await save_sentiment(
                        conn,
                        run_id=self.run_id,
                        input_hash=identities[request_id],
                        article_set_id=article_set_id,
                        market_id=market.market_id,
                        pass_number=row["pass_number"],
                        symbol=asset.symbol,
                        provider="ollama",
                        model_name=self.ollama.model_name,
                        prompt_version=self.config.ollama_sentiment_prompt_version,
                        model_input=batch_model_input,
                        model_output=batch_model_output,
                        result=result,
                    )
        for row, _, asset, _, articles in prepared:
            work_key = f"{row['market_id']}:{row['pass_number']}:{asset.symbol}"
            await finish_work(
                conn,
                run_id=self.run_id,
                stage="research",
                work_key=work_key,
                result={"article_count": len(articles)},
            )
        print(f"[research batch {batch_index}] jobs={len(prepared)}")
