


class IB_Connect:
    def __init__(self, host, port, client_id):
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self):
        pass

    def disconnect(self):
        pass


    def get_account_summary(self):
        pass


    def get_positions(self):
        pass

    def place_order(self, action, symbol, quantity, order_type):
        pass

