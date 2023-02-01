import pytest
from environment.Env import Environment
from exchange_api.coinbase_api import ExchangeAPI, BalanceSpec
from datetime import datetime,timedelta

class TestApi(ExchangeAPI):
    def __init__(self) -> None:
        self.available = self.balance = {
            "USD": 1.0,
            "BTC": 1.0
            }
        
    def updateAvail(self) -> None:
        return
    
    def updateBalance(self) -> None:
        return

class TestEnv:

    def __init__(self):
        api = self.test_api = self.getTestApi()
        start_time = datetime.now()
        end_time = start_time + timedelta(hours = 1)
        products = ["ABC"]
        
        self.env = Environment(start_time, end_time, api, products)
    def getTestApi(self) -> TestApi:
        pass

    def testSetup(self) -> None:
        """
        Setup should be able to given an api return starting state. 
        I think I need to clean up this method before this test makes sense,
        but for now I will just say what I expect the test to do. 
        """

        env = self.env

        expected_state = [[0.0  for _ in range(7)]]

        state = env.setup()

        assert expected_state == state 

    def testStepEnv(self) -> None: 
        """
        Step should return state, reward and done.
        Probably also need to clean up this method. 
        """
        env = self.env 

        expected_state = [[0.0  for _ in range(7)]]

        expected_reward = [[0]]

        expected_done = 0

        state, reward, done = env.step()

        assert state == expected_state

        assert reward == expected_reward

        assert done == expected_done

    def testAct(self) -> None:
        """
        Given a set of actions act should delegate the actions to their respective
        product features.
        """
        env = self.env

        actions = [[0.0, 0.0]]

        env.act(actions) # Currently this has no return val, will change.

class TestProductFeatures:

    pass