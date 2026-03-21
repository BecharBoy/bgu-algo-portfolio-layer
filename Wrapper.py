import asyncio
from concurrent.futures import ThreadPoolExecutor

class Wrapper:

    def __init__(self):
        pass

    async def run_cointegration_scan(self, price_matrix, num_threads: int = 4, threshold: float = 0.85) -> list:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            pass




