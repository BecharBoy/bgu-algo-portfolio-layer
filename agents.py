


class agents:

    def __init__(self):
        # TODO: Inject Gemini client wrapper and prompt templates.
        # TODO: Add guardrails so agents cannot trigger execution side effects.
        self.model_name = "gemini-2.5-flash"


    def insights_from_trades_agent(self):
        # TODO: Input contract should include trades, fills, and signal context.
        # TODO: Return structured insights for API/website consumption.
        # TODO: Add retry and timeout policy for LLM calls.
        pass


    def accountant_agent(self):
        """ an agent that job is to make sure our risk, max drawdown and managing the investments is in a right place"""
        # TODO: Input contract should include account snapshots and open risk.
        # TODO: Output should contain warnings + suggested risk actions.
        # TODO: Keep recommendations read-only until explicit approval flow exists.
        pass

    def build_agent_context(self, *, trades: list[dict], signals: list[dict], account_snapshot: dict) -> dict:
        # TODO: Normalize raw DB payloads into one context object for all agents.
        # TODO: Include timestamps and strategy-level aggregates.
        # TODO: Add token-budget trimming for long histories.
        pass
