import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import pandas as pd

class Wrapper:
    def __init__(self, num_threads: int = 4, min_correlation: float = 0.85):
        self.num_threads = num_threads
        self.min_correlation = min_correlation # a high bound on the correlations that we think are interesting
        """ add the compiled wrapper of the c++ code we are gonna wrap in bind - ensures us only after we compile the
        cpp, the cointegration_engine will be called """
        import cointegration_engine
        self.cointegration_engine = cointegration_engine
        # TODO: Lazily import compiled cointegration module and validate availability.

    def _prepare_matrix(self, market_data: Dict[str, pd.DataFrame]) -> Tuple[List[str], List[List[float]]]:
        """
        TODO: Extract tickers and their 'Close' prices into a flat list of names
        and a 2D list of floats for C++.
        """
        # TODO: Align all ticker series by date and handle missing values consistently.
        # TODO: Validate minimum lookback and equal-length arrays for C++ scanner.
        """ preparing the data for the cpp functions, in math stats we except too get a matrix so we need to make sure,
        the matrix we delivering have all the stocks have exactly the same trading days """
        if not market_data:
            return [], []

        close_prices = {ticker:df['Close'] for ticker, df in market_data.items()}
        aligned_df = pd.DataFrame(close_prices).dropna()

        if aligned_df.empty:
            return [], []

        tickers = aligned_df.columns.tolist()
        price_matrix = aligned_df.values.T.tolist()
        return tickers, price_matrix

    async def run_cointegration_scan(self, market_data: Dict[str, pd.DataFrame]) -> list:
        """
        TODO:
        1. Call _prepare_matrix
        2. Use asyncio.get_running_loop().run_in_executor() with a ThreadPoolExecutor
        3. Execute the C++ module function (cointegration_engine.run_cpp_scan)
        4. Return the results
        """
        """ using Threadpool to run "_run_cpp_scan_in_executor we have in bindings with the matrix we aligned"""
        if self.cointegration_engine is None:
            raise RuntimeError('Cannot run scan: cointegration_engine is not loaded.')
        tickers, price_matrix = self._prepare_matrix(market_data)
        if len(tickers) < 2 or len(price_matrix[0]) < 3:
            return []

        results = await self._run_cpp_scan_in_executor(tickers, price_matrix)
        return results

    async def _run_cpp_scan_in_executor(self, tickers: List[str], price_matrix: List[List[float]]) -> list:
        # TODO: Use run_in_executor(ThreadPoolExecutor) to call C++ extension safely.
        # TODO: Map raw C++ results into Python dictionaries/dataclasses.
        """ _run_cpp_scan_in_executor return CointegratedPair which we need to break and parse
         it to a more readable result """
        loop = asyncio.get_running_loop()

        with ThreadPoolExecutor(max_workers=1) as pool:
            raw_results = await loop.run_in_executor(
                pool,
                self.cointegration_engine.run_cpp_scan,
                tickers,
                price_matrix,
                self.num_threads,
                self.min_correlation
            )
        parsed_results = []
        for res in raw_results:
            parsed_results.append({
                "stock_x": res.stock_x,
                "stock_y": res.stock_y,
                "correlation": res.correlation,
                "hedge_ratio": res.hedge_ratio,
                "adf_stat": res.adf_stat
            })
        return parsed_results
