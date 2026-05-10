import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class Wrapper:
    def __init__(self, num_threads: int = 4, min_correlation: float = 0.85):
        self.num_threads = num_threads
        self.min_correlation = min_correlation
        import cointegration_engine
        self.cointegration_engine = cointegration_engine
        # Executor is created once and reused for every scan call.
        # max_workers=1 is intentional: C++ manages its own internal thread pool.
        # The single Python thread just dispatches the blocking call and releases the GIL.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _prepare_matrix(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Extract tickers and their Close prices into a flat 2D list for C++.
        Aligns all series by date. Drops tickers with insufficient history after alignment.
        """
        if not market_data:
            return [], []

        close_prices = {ticker: df["Close"] for ticker, df in market_data.items()}
        aligned_df = pd.DataFrame(close_prices).dropna()

        if aligned_df.empty:
            return [], []

        tickers = aligned_df.columns.tolist()
        price_matrix = aligned_df.values.T.tolist()
        return tickers, price_matrix

    async def run_cointegration_scan(self, market_data: Dict[str, pd.DataFrame]) -> list:
        if self.cointegration_engine is None:
            raise RuntimeError("Cannot run scan: cointegration_engine is not loaded.")
        tickers, price_matrix = self._prepare_matrix(market_data)
        if len(tickers) < 2 or len(price_matrix[0]) < 3:
            return []
        results = await self._run_cpp_scan_in_executor(tickers, price_matrix)
        return results

    async def _run_cpp_scan_in_executor(
        self, tickers: List[str], price_matrix: List[List[float]]
    ) -> list:
        loop = asyncio.get_running_loop()
        raw_results = await loop.run_in_executor(
            self._executor,
            self.cointegration_engine.run_cpp_scan,
            tickers,
            price_matrix,
            self.num_threads,
            self.min_correlation,
        )
        return [
            {
                "stock_x":    res.stock_x,
                "stock_y":    res.stock_y,
                "correlation": res.correlation,
                "hedge_ratio": res.hedge_ratio,
                "adf_stat":    res.adf_stat,
            }
            for res in raw_results
        ]
