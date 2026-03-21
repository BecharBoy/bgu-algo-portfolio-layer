from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any


class BaseStrategy(ABC):
    def __init__(self, name: str, weight_allocation: float):
        self.name = name
        self.weight_allocation = weight_allocation
        # TODO: Add optional per-strategy config payload.

    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame], current_positions: Dict[str, Any]) -> List[dict]:
        # TODO: Return normalized signal schema shared by all strategies.
        pass

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        # TODO: Centralize basic signal schema validation in base class.
        # TODO: Consider moving to shared schema module later.
        pass


