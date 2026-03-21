

class Portfolio:
    def __init__(self, ib_client, db_client):
        self.ib_client = ib_client
        self.db_client = db_client
        self.strategies = {}

    def add_strategy(self, name, strategy):
        self.strategies[name] = strategy


    async def run_cycle(self):
        pass


    def allocate_capital(self, signals):
        pass


    def excecute_approved_signals(self):
        pass

