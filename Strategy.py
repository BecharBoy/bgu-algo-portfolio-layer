from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, name: str, weight_allocation: float):
        self.name = name
        self.weight_allocation = weight_allocation

    @abstractmethod
    async def generate_signals(self, market_data: pd.dataFrame) -> list[dict]:
        pass


