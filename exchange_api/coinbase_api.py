import requests
import math
import json, hmac, hashlib, time, requests, base64
from requests.auth import AuthBase

# Create custom authentication for Exchange


class CoinbaseExchangeAuth(AuthBase):
    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase

    def __call__(self, request):

        timestamp = str(time.time())
        message = (
            timestamp
            + request.method
            + request.path_url
            + (request.body or b"").decode("utf-8")
        )
        hmac_key = base64.b64decode(self.secret_key)
        signature = hmac.new(hmac_key, message.encode("utf-8"), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest())

        request.headers.update(
            {
                "CB-ACCESS-SIGN": signature_b64,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-PASSPHRASE": self.passphrase,
                "Content-Type": "application/json",
            }
        )
        return request


class CoinbaseAPI:
    def __init__(self, api_url, api_key, secret_key, passphrase):
        self.api_url = api_url
        self.auth = CoinbaseExchangeAuth(api_key, secret_key, passphrase)
        self.available = self.getAvail()
        self.balance = self.getBalance()

    def getAvail(self):
        r = requests.get(self.api_url + "accounts", auth=self.auth)
        return {item["currency"]: float(item["available"]) for item in r.json()}

    def getBalance(self):
        r = requests.get(self.api_url + "accounts", auth=self.auth)

        return {item["currency"]: float(item["balance"]) for item in r.json()}

    def placeOrder(self, size, price, side, product):
        r = requests.post(
            self.api_url + "orders",
            json={
                "size": size,
                "price": price,
                "side": side,
                "product_id": product,
                "stp": "cb",
                "time_in_force": "GTT",
                "cancel_after": "min",
            },
            auth=self.auth,
        )
        return r.json()

    def cancelAllOrders(self):
        r = requests.delete(self.api_url + "orders", auth=self.auth)
        return r.json()

    def getOrderBook(self, product, level=2):
        r = requests.get(
            self.api_url + "products/{}/book?level={}".format(product, level),
            auth=self.auth,
        )
        return r.json()

    def getHistoric(self, product, granularity=86400, start=None, end=None):
        if not start or not end:
            r = requests.get(
                self.api_url
                + "products/{}/candles?granularity={}".format(product, granularity),
                auth=self.auth,
            )
        else:
            r = requests.get(
                self.api_url
                + "products/{}/candles?granularity={}&start={}&end={}".format(
                    product, granularity, start, end
                ),
                auth=self.auth,
            )

        return r.json()

    def getBaseQuote(self, product):
        r = requests.get(self.api_url + "products/{}".format(product), auth=self.auth)
        return r.json()["base_increment"].count("0"), r.json()["quote_increment"].count(
            "0"
        )

    def updateAvail(self):
        self.available = self.getAvail()

    def updateBalance(self):
        self.balance = self.getBalance()

    def getTrades(self, product):
        r = requests.get(
            self.api_url + "products/{}/trades".format(product), auth=self.auth
        )
        return r.json()

    def cancelOrder(self, order):
        r = requests.delete(self.api_url + "orders/{}".format(order), auth=self.auth)
        return r.json()

    def getOrder(self, order):
        r = requests.get(self.api_url + "orders/{}".format(order), auth=self.auth)
        return r.json()

    def getFills(self, product, before):
        r = requests.get(
            self.api_url + "products/{}/trades?before={}".format(product, before)
        )
        # print(r.headers)
        return r.json(), r.headers["cb-before"]

    def getProductNames(self):
        r = requests.get(self.api_url + "products", auth=self.auth)

        return [
            item["id"]
            for item in r.json()
            if not (
                item["trading_disabled"]
                and item["cancel_only"]
                and item["post_only"]
                and item["limit_only"]
            )
        ]

    def getFees(self):
        r = requests.get(self.api_url + "fees", auth=self.auth)
        return r.json()
