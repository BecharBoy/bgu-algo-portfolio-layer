from __future__ import annotations

from typing import Any
from database.backtesting.repositories.runs import finish_work, start_work


async def run(self, conn: Any) -> None:
    self.current_work_key = "gdelt"
    if not await start_work(
        conn,
        run_id=self.run_id,
        stage="preflight",
        work_key=self.current_work_key,
        payload={
            "service": "GDELT",
            "minimum_request_interval_seconds": (
                self.config.gdelt_minimum_request_interval_seconds
            ),
        },
    ):
        return
    article_count = await self.news.healthcheck(
        as_of=min(self.config.end, self.config.historical_data_cutoff)
    )
    await finish_work(
        conn,
        run_id=self.run_id,
        stage="preflight",
        work_key=self.current_work_key,
        result={"service": "GDELT", "sample_article_count": article_count},
    )
