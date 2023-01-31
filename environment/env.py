from pathlib import Path
import datetime
import torch
from collections import deque


import numpy as np

useSandBox = False

import torch


class Enviroment:
    def __init__(
        self, start_time, end_time, api, products, session_length=3600, penalty=0.005
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.api = api
        self.products = products
        self.product_mapping = {product: i for i, product in enumerate(products)}
        self.product_features = {}
        self.penalty = penalty
        self.session_length = session_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.oldTotal = []
        self.gamma = 0.3
        self.pos_pnl = 0

    def setup_env(self):
        self.api.updateAvail()
        self.api.updateBalance()
        fees = self.api.getFees()
        makerFee, takerFee = float(fees["maker_fee_rate"]), float(
            fees["taker_fee_rate"]
        )

        self.startTime = self.lastUpdateTime = datetime.datetime.now()
        self.end_time
        state = [[] for _ in range(len(self.products))]
        self.oldTotal = self.api.balance["USD"]
        for item in np.random.permutation(len(self.products)):
            product = self.products[item]

            productFeatures = ProductFeatures(
                product,
                api,
                makerFee,
                takerFee,
                self.startTime,
                sessionLength=self.session_length,
            )

            self.product_features[product] = productFeatures

            sellPos = productFeatures.sellBalance * (
                productFeatures.midPoint * productFeatures.botPrice
            )

            self.oldTotal += sellPos

            productState = productFeatures.getProductFeatures()

            state[item] = productState[::]
        self.startingTotal = self.oldTotal
        return state

    def stepEnv(self):
        self.api.updateAvail()
        self.api.updateBalance()
        fees = self.api.getFees()
        makerFee, takerFee = float(fees["maker_fee_rate"]), float(
            fees["taker_fee_rate"]
        )

        newUpdateTime = datetime.datetime.now()
        done = 0

        if (newUpdateTime - self.startTime).total_seconds() > self.session_length:
            done = 1

        timeDelta = (
            newUpdateTime - self.lastUpdateTime
        ).total_seconds() / self.session_length
        state = [[] for _ in range(len(self.products))]

        reward = [0 for i in range(len(self.products))]
        newTotal = self.api.balance["USD"]
        flag = False
        if timeDelta * self.session_length > 180:
            flag = True

        for item in np.random.permutation(len(self.products)):
            product = self.products[item]

            productFeatures = self.product_features[product]

            new_state, posPnl, can_update = productFeatures.update(
                makerFee, takerFee, newUpdateTime, api
            )
            flag = flag or can_update
            self.product_features[product] = productFeatures
            self.pos_pnl = posPnl * self.gamma + (1 - self.gamma) * self.pos_pnl
            newTotal += (
                productFeatures.sellBalance
                * productFeatures.midPoint
                * productFeatures.botPrice
            )
            state[item] = new_state[::]

        self.lastUpdateTime = newUpdateTime
        n = len(self.products)
        log_returns = np.log(newTotal) - np.log(self.oldTotal)
        reward = log_returns - posPnl

        self.oldTotal = newTotal

        return state, [reward for _ in range(n)], [done for _ in range(n)]

    def act(self, actions):

        for i, action in enumerate(actions):

            product = self.products[i]

            productFeatues = self.product_features[product]

            productFeatues.act(action)


class ProductFeatures:
    def __init__(
        self, product, api, makerFee, takerFee, startTime, gamma=0.9, sessionLength=3600
    ):
        self.gamma = gamma
        self.rho = 0.3
        self.api = None
        self.product = product
        self.ewma = self.midPoint = None
        self.bidVol = None
        self.askVol = None
        self.buySize = None
        self.sellSize = None
        self.buyBalance = None
        self.sellBalance = None
        self.base = None
        self.quote = None
        self.quoteInc = None
        self.spread = None
        self.makerFee = None
        self.takerFee = None
        self.updateTime = None
        self.startTime = startTime
        self.avgPrice = None
        self.tradingVolume = None
        self.endTime = self.startTime + datetime.timedelta(seconds=sessionLength)
        self.timeRemaining = self.sessionLength = sessionLength
        self.lamb = 0
        self.positionalPnl = None
        self.ewmaVar = 0
        self.orders = deque()
        self.vwop_bid = 0
        self.vwop_ask = 0
        self.starting_cash = 0.5
        self.starting_stock = 0.5
        self.update(makerFee, takerFee, startTime, api)

    def update(self, makerFee, takerFee, updateTime, api):
        p1, p2 = self.product.split("-")
        self.api = api
        self.makerFee = makerFee
        self.takerFee = takerFee
        bids, asks = self.get_bid_ask(self.product, 3)
        bestBid, bestAsk = float(bids[0][0]), float(asks[0][0])
        if self.midPoint is None:
            self.startingPoint = oldMidPoint = midPoint = self.midPoint = (
                bestAsk + bestBid
            ) / 2
        else:
            oldMidPoint = self.midPoint
            midPoint = self.midPoint = (bestAsk + bestBid) / 2
        self.updateTime = updateTime
        self.timeRemaining = (self.endTime - updateTime).total_seconds()
        if self.ewma is None:
            self.ewma = midPoint

        self.ewma = self.rho * midPoint + (1 - self.rho) * self.ewma

        if self.avgPrice is None:
            self.avgPrice = midPoint
            self.tradingVolume = 0

        self.ewmaVar = (
            self.rho * (midPoint - self.avgPrice) ** 2 + (1 - self.rho) * self.ewmaVar
        )

        bidVol, askVol = sum(float(bids[i][1]) for i in range(len(bids))), sum(
            float(asks[i][1]) for i in range(len(asks))
        )
        if self.bidVol is None:
            self.bidVol = bidVol
            self.askVol = askVol
        self.bidVol = self.rho * bidVol + (1 - self.rho) * self.bidVol
        self.askVol = self.rho * askVol + (1 - self.rho) * self.askVol
        if self.base is None:
            base, quote = api.getBaseQuote(self.product)
            self.base = base
            self.quote = quote
            self.quoteInc = 10 ** (-quote)

        self.spread = (bestAsk - bestBid) / self.quoteInc
        self.bestBid = bestBid
        self.bestAsk = bestAsk
        self.buyBalance, self.sellBalance = api.balance[p2], api.balance[p1]
        self.buySize, self.sellSize = api.available[p2], api.available[p1]
        if self.positionalPnl is None:
            self.positionalPnl = (self.midPoint) * self.sellBalance + self.buyBalance
        oldPnl = self.positionalPnl
        self.positionalPnl = (self.midPoint) * self.sellBalance + self.buyBalance
        if p2 == "USD":
            botPrice = 1
        else:
            botBid, botAsk = self.get_bid_ask("{}-USD".format(p2), 5)
            botBid, botAsk = float(botBid[0][0]), float(botAsk[0][0])
            botPrice = (botBid + botAsk) / 2
        self.botPrice = botPrice
        flag = False
        if not self.orders:
            flag = True

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

                volumeTop = float(r["filled_size"])
                price = float(r["executed_value"]) / volumeTop
                val = float(r["executed_value"]) * botPrice
                fee = float(r["fill_fees"])
                flag = True
                quoted_val = float(r["price"])
                quoted_size = float(r["size"])
                if r["side"] == "buy":

                    self.avgPrice = (
                        volumeTop * (price) + self.avgPrice * self.tradingVolume
                    ) / (volumeTop + self.tradingVolume)
                    self.tradingVolume += volumeTop

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
                            (midPoint - quoted_val) / self.quoteInc * quoted_size
                        )
                        buy_vol += quoted_size
                    if r["side"] == "sell":
                        self.vwop_ask += (
                            (quoted_val - midPoint) / self.quoteInc * quoted_size
                        )
                        sell_vol += quoted_size
            elif "status" in r and r["status"] == "open":
                self.orders.appendleft(order)
            else:
                api.cancelOrder(order)
        self.vwop_ask /= sell_vol + np.finfo(float).eps
        self.vwop_bid /= buy_vol + np.finfo(float).eps
        old_total = self.starting_cash + self.starting_stock
        new_total = self.starting_stock * midPoint / oldMidPoint + self.starting_cash
        self.starting_cash = self.starting_stoct = new_total / 2
        return self.getProductFeatures(), np.log(new_total) - np.log(old_total), flag

    def get_bid_ask(self, product, depth=None):

        book = self.api.getOrderBook(product)

        if depth is None:
            bids = book["bids"]
            asks = book["ask"]
        else:
            bids = book["bids"][0:depth]
            asks = book["asks"][0:depth]

        return bids, asks

    def getProductFeatures(self):
        midPrice = self.midPoint
        botPrice = self.botPrice
        normalizedBidVol = self.bidVol * (midPrice * botPrice)
        normalizedAskVol = self.askVol * (midPrice * botPrice)
        normalizedBuySize = self.buySize * (botPrice)
        normalizedSellSize = self.sellSize * (midPrice * botPrice)
        normalizedBuyBal = self.buyBalance * (botPrice)
        normalizedSellBal = self.sellBalance * (midPrice * botPrice)
        normalizedLamb = self.lamb / self.sessionLength
        normalizedTimeRemaining = self.timeRemaining / self.sessionLength
        bid_dispersion = float(self.vwop_bid) * (normalizedBuyBal - normalizedBuySize)
        ask_dispersion = float(self.vwop_ask) * (normalizedSellBal - normalizedSellSize)
        ord_dis = (bid_dispersion - ask_dispersion) / (
            normalizedBuyBal + normalizedSellBal
        )
        out = [
            100 * 100 * (midPrice - self.avgPrice) / self.startingPoint,
            10
            * (normalizedBuyBal - normalizedSellBal)
            / (normalizedBuyBal + normalizedSellBal),
            ord_dis,
            100 * 100 * (self.ewma - self.avgPrice) / self.startingPoint,
            100 * 100 * (self.ewmaVar) / (self.startingPoint * self.startingPoint),
            10 * normalizedTimeRemaining,
            10
            * (normalizedAskVol - normalizedBidVol)
            / (normalizedBidVol + normalizedAskVol),
        ]
        return out

    def act(self, action):
        print(action)
        bidOffSet, askOffSet = action

        bid = round(self.bestBid - bidOffSet * self.quoteInc, self.quote)
        ask = round(self.bestAsk + askOffSet * self.quoteInc, self.quote)

        buySize = min(
            0.05 * (self.buyBalance / self.midPoint + self.sellBalance),
            0.1 * self.buySize / self.midPoint,
        )
        buySize = round(buySize, self.base)
        sellSize = min(
            0.05 * (self.buyBalance / self.midPoint + self.sellBalance),
            0.1 * self.sellSize,
        )
        sellSize = round(sellSize, self.base)

        buyOrder = self.api.placeOrder(buySize, bid, "buy", self.product)
        sellOrder = self.api.placeOrder(sellSize, ask, "sell", self.product)

        if "id" in buyOrder:
            self.orders.append(buyOrder["id"])

        if "id" in sellOrder:
            self.orders.append(sellOrder["id"])
