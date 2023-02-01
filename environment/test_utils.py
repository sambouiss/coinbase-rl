from exchange_api.coinbase_api import ExchangeAPI, BalanceSpec, OrderSpec
from typing import List


class TestApi(ExchangeAPI):
    def __init__(self) -> None:
        self.available = self.balance = {
            "USD": 1.0,
            "BTC": 1.0
            }
        
    def updateAvail(self) -> None:
        return self.available
    
    def updateBalance(self) -> None:
        return self.balance
    
    def cancelAllOrders(self) -> List[str]:
        return []
    
    def cancelOrder(self, order: str) -> object:
        return {}
    
    def getAvail(self) -> BalanceSpec:
        return self.available
    
    def getBalance(self) -> BalanceSpec:
        return self.balance
    
    def getBaseQuote(self, product: str) -> object:
        return 1.0, 1.0
    
    def getFills(self, product: str, before: int) -> object:
        return []
    
    def getFees(self) -> object:
        return {}
    
    def getHistoric(self, product: str, granularity: int, start: str, end: str) -> object:
        return []
    
    def getOrderBook(self, product: str, level: int) -> object:
        return {}
    
    def getProductNames(self) -> List[str]:
        return ["BTC"]
    
    def getTrades(self, product: str) -> object:
        return  []
    
    def placeOrder(self, size: float, price: float, side: str, product: str) -> OrderSpec:
        return {}