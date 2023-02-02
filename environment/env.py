from datetime import datetime,timedelta
import torch
from collections import deque
from typing import List, NamedTuple, Tuple
from exchange_api.coinbase_api import CoinbaseAPI
import numpy as np
from abc import ABC, abstractmethod, abstractproperty

useSandBox = False

import torch

State = List[float]

"""
I think the dependence of environment on product features kinda points to an issue with
design that I need to fix.

I think in the long run maybe using wrappers to make it so api info is accessed in a uniform
would be a good improvement.
"""
class ProductFeatures:
    def __init__(
        self, 
        product: str, 
        api: CoinbaseAPI, 
        maker_fee: float, 
        taker_fee: float,
        start_time: datetime, 
        gamma: float=0.9, 
        session_length: int=3600
    ) -> None:
        self.gamma = gamma
        self.rho = 0.3
        self.api = None
        self.product = product
        self.ewma = None
        self.mid_point = None
        self.bid_vol = None
        self.ask_vol = None
        self.buy_size = None
        self.sell_size = None
        self.buy_balance = None
        self.sell_balance = None
        self.base = None
        self.quote = None
        self.quote_inc = None
        self.spread = None
        self.maker_fee = None
        self.taker_fee = None
        self.update_time = None
        self.start_time = start_time
        self.avg_price = None
        self.trading_volume = None
        self.end_time = self.start_time + datetime.timedelta(seconds=session_length)
        self.time_remaining = session_length
        self.session_length = session_length
        self.lamb = 0
        self.positional_pnl = None
        self.ewma_var = 0
        self.orders = deque()
        self.vwop_bid = 0
        self.vwop_ask = 0
        self.starting_cash = 0.5
        self.starting_stock = 0.5
        self.update(maker_fee, taker_fee, start_time, api)

    def update(self, maker_fee, taker_fee, update_time, api, set_start_time = False):
        if set_start_time:
            self.start_time = datetime.now()
            self.end_time = self.start_time+timedelta(seconds=self.session_length)
        p1, p2 = self.product.split("-")
        self.api = api
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        bids, asks = self.getBidAsk(3)
        best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
        if self.mid_point is None:
            mid_point = (
                best_ask + best_bid
            ) / 2
            old_mid_point = mid_point
            self.starting_price = mid_point
            self.mid_point = mid_point
        else:
            mid_point = (best_ask + best_bid) / 2
            old_mid_point = self.mid_point
            self.mid_point = mid_point
        self.update_time = update_time
        self.time_remaining = (self.end_time - update_time).total_seconds()
        if self.ewma is None:
            self.ewma = mid_point

        self.ewma = self.rho * mid_point + (1 - self.rho) * self.ewma

        if self.avg_price is None:
            self.avg_price = mid_point
            self.trading_volume = 0

        self.ewma_var = (
            self.rho * (mid_point - self.avg_price) ** 2 + (1 - self.rho) * self.ewma_var
        )

        bid_vol, ask_vol = sum(float(bids[i][1]) for i in range(len(bids))), sum(
            float(asks[i][1]) for i in range(len(asks))
        )
        if self.bid_vol is None:
            self.bid_vol = bid_vol
            self.ask_vol = ask_vol
        self.bid_vol = self.rho * bid_vol + (1 - self.rho) * self.bid_vol
        self.ask_vol = self.rho * ask_vol + (1 - self.rho) * self.ask_vol
        if self.base is None:
            base, quote = api.getBaseQuote(self.product)
            self.base = base
            self.quote = quote
            self.quote_inc = 10 ** (-quote)

        self.spread = (best_ask - best_bid) / self.quote_inc
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.buy_balance, self.sell_balance = api.balance[p2], api.balance[p1]
        self.buy_size, self.sell_size = api.available[p2], api.available[p1]
        if self.positional_pnl is None:
            self.positional_pnl = (self.mid_point) * self.sell_balance + self.buy_balance
        old_pnl = self.positional_pnl
        self.positional_pnl = (self.mid_point) * self.sell_balance + self.buy_balance
        if p2 == "USD":
            bot_price = 1
        else:
            bot_bid, bot_ask = self.get_bid_ask("{}-USD".format(p2), 5)
            bot_bid, bot_ask = float(bot_bid[0][0]), float(bot_ask[0][0])
            bot_price = (bot_bid + bot_ask) / 2
        self.bot_price = bot_price
        

        sell_vol = 0
        buy_vol = 0
        self.vwop_bid = self.vwop_ask = 0
        while len(self.orders) > 10:
            order = self.orders.popleft()
            api.cancelOrder(order)
        for _ in range(len(self.orders)):
            order = self.orders.pop()
            r = api.getOrder(order)
            if "status" in r and r["status"] == "done" and r["done_reason"] == "filled":
                fillTime = datetime.datetime.utcnow() - datetime.datetime.fromisoformat(
                    r["created_at"].split(".")[0]
                )
                fillTime = fillTime.total_seconds()
                self.lamb = self.rho * fillTime + (1 - self.rho) * self.lamb

                volume_top = float(r["filled_size"])
                price = float(r["executed_value"]) / volume_top
                val = float(r["executed_value"]) * bot_price
                fee = float(r["fill_fees"])
                
                quoted_val = float(r["price"])
                quoted_size = float(r["size"])
                if r["side"] == "buy":

                    self.avg_price = (
                        volume_top * (price) + self.avg_price * self.trading_volume
                    ) / (volume_top + self.trading_volume)
                    self.trading_volume += volume_top

            elif "created_at" in r and "status" in r and r["status"] == "open":
                fillTime = datetime.datetime.utcnow() - datetime.datetime.fromisoformat(
                    r["created_at"].split(".")[0]
                )
                fillTime = fillTime.total_seconds()
                if fillTime > 600:
                    api.cancelOrder(order)
                else:
                    self.orders.appendleft(order)
                    quoted_val = float(r["price"])
                    quoted_size = float(r["size"])
                    if r["side"] == "buy":

                        self.vwop_bid += (
                            (mid_point - quoted_val) / self.quote_inc * quoted_size
                        )
                        buy_vol += quoted_size
                    if r["side"] == "sell":
                        self.vwop_ask += (
                            (quoted_val - mid_point) / self.quote_inc * quoted_size
                        )
                        sell_vol += quoted_size
            elif "status" in r and r["status"] == "open":
                self.orders.appendleft(order)
            else:
                api.cancelOrder(order)
        self.vwop_ask /= sell_vol + np.finfo(float).eps
        self.vwop_bid /= buy_vol + np.finfo(float).eps
        old_total = self.starting_cash + self.starting_stock
        new_total = self.starting_stock * mid_point / old_mid_point + self.starting_cash
        self.starting_cash = self.starting_stock = new_total / 2
        return self.getProductState(), np.log(new_total) - np.log(old_total)

    def getBidAsk(self, depth=None):

        book = self.api.getOrderBook(self.product)

        if depth is None:
            bids = book["bids"]
            asks = book["ask"]
        else:
            bids = book["bids"][0:depth]
            asks = book["asks"][0:depth]

        return bids, asks
    
    @property
    def stock_position(self) -> float:
        return self.sell_balance * self.mid_point * self.bot_price
    
    @property
    def cash_position(self) -> float:
        return self.sell_balance * self.bot_price
    
    def getProductState(self):
        mid_point = self.mid_point
        bot_price = self.bot_price

        normalized_bid_vol = self.bid_vol * (mid_point * bot_price)
        normalized_ask_vol = self.ask_vol * (mid_point * bot_price)

        normalized_buy_size = self.buy_size * (bot_price)
        normalized_sell_size = self.sell_size * (mid_point * bot_price)

        nomralized_buy_balance = self.cash_position
        normalized_sell_balance = self.stock_position
        
        starting_point = self.starting_point

        normalized_time_remaining = self.time_remaining / self.session_length

        bid_dispersion = float(self.vwop_bid) * (nomralized_buy_balance - normalized_buy_size)
        ask_dispersion = float(self.vwop_ask) * (normalized_sell_balance - normalized_sell_size)

        ord_dispersion = (bid_dispersion - ask_dispersion) 
        ord_dispersion /= nomralized_buy_balance + normalized_sell_balance
        

        position =  10 * (nomralized_buy_balance - normalized_sell_balance)
        position /= (nomralized_buy_balance + normalized_sell_balance)

        order_imbalance = (normalized_ask_vol - normalized_bid_vol)
        order_imbalance /= (normalized_bid_vol + normalized_ask_vol)

        price_dispersion =  100 * 100 * (mid_point - self.avg_price)  
        price_dispersion /= starting_point 

        ewma_dispersion = 100 * 100 * (self.ewma - self.avg_price)
        ewma_dispersion /= starting_point
        
        normalized_var =  100 * 100 * (self.ewma_var)
        normalized_var /= (starting_point * starting_point)
        
        out = [
            price_dispersion,
            position,
            ord_dispersion,
            ewma_dispersion,
            normalized_var,
            10 * normalized_time_remaining,
            10 * order_imbalance,
        ]
        return out

    def act(self, action):
        print(action)
        bid_offset, ask_offset = action

        bid = round(self.best_bid - bid_offset * self.quote_inc, self.quote)
        ask = round(self.best_ask + ask_offset * self.quote_inc, self.quote)

        buy_size = min(
            0.05 * (self.buy_balance / self.mid_point + self.sell_balance),
            0.1 * self.buy_size / self.mid_point,
        )
        buy_size = round(buy_size, self.base)
        sell_size = min(
            0.05 * (self.buy_balance / self.mid_point + self.sell_balance),
            0.1 * self.sell_size,
        )
        sell_size = round(sell_size, self.base)

        buy_order = self.api.placeOrder(buy_size, bid, "buy", self.product)
        sell_order = self.api.placeOrder(sell_size, ask, "sell", self.product)

        if "id" in buy_order:
            self.orders.append(buy_order["id"])

        if "id" in sell_order:
            self.orders.append(sell_order["id"])
            
class Environment:
    def __init__(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        api: CoinbaseAPI, 
        products: str, 
        product_features: List[ProductFeatures],
        session_length: int = 3600, 
        penalty: float = 0.005,
    ) -> None:
        assert product_features
        self.start_time = start_time
        self.end_time = end_time
        self.api = api
        self.products = products
        self.product_features = product_features
        self.product_names = set()
        for productFeatures in product_features:
            p1, p2 = productFeatures.product.split("-")
            self.product_names.add(p1)
            self.product_names.add(p2)
        self.penalty = penalty
        self.session_length = session_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.old_total = 0
        self.gamma = 0.3
        self.pos_pnl = 0

    def setup(self) -> List[State]:
        api = self.api
        api.updateAvail()
        api.updateBalance()

        fees = api.getFees()

        maker_fee = float(fees["maker_fee_rate"])
        taker_fee = float(fees["taker_fee_rate"])

        self.start_time = self.last_update_time = datetime.now()
        self.end_time = self.start_time+timedelta(seconds=self.session_length)
        state = []
        self.old_total = self.api.balance["USD"]
        
        for productFeatures in self.product_features:
            product_state, _ = productFeatures.update(maker_fee,taker_fee,self.last_update_time,api,True)
            
            state.append(product_state)
            
            self.old_total += productFeatures.stock_position
            
        self.starting_total = self.old_total
        
        return state

    def step(self) -> Tuple(List[State], List[float], List[int]):
        api = self.api
        api.updateAvail()
        api.updateBalance()
        
        fees = api.getFees()
        
        maker_fee = float(fees["maker_fee_rate"])
        taker_fee = float(fees["taker_fee_rate"])
        
        new_update_time = datetime.now()
        done = 0

        if (new_update_time - self.start_time).total_seconds() > self.session_length:
            done = 1

        state = []

        reward = [0 for _ in range(len(self.products))]
        new_total = self.api.balance["USD"]
        

        for productFeatures in self.product_features:
           

            new_state, pos_pnl = productFeatures.update(
                maker_fee, taker_fee, new_update_time, api
            )
            
            
            self.pos_pnl = pos_pnl * self.gamma + (1 - self.gamma) * self.pos_pnl
            new_total += productFeatures.stock_position
            state.append(new_state)

        self.last_update_time = new_update_time
        
        n = len(self.products)
        log_returns = np.log(new_total) - np.log(self.old_total)
        reward = log_returns - pos_pnl

        self.old_total = new_total

        return state, [reward for _ in range(n)], [done for _ in range(n)]

    def act(self, actions):

        for i, action in enumerate(actions):

            productFeatues = self.product_features[ i]

            productFeatues.act(action)



