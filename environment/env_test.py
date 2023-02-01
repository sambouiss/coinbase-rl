import pytest
from environment.Env import Environment
from environment.test_utils import TestApi
from datetime import datetime,timedelta
from typing import List

class TestEnv:

    def getTestApi(self) -> TestApi:
        return TestApi()

    def getTestEnv(self) -> Environment:
        api = self.getTestApi()
        start_time = datetime.now()
        end_time = start_time + timedelta(hours = 1)
        products = ["BTC"]
        return Environment(start_time, end_time, api, products)

    def testSetup(self) -> None:
        """
        Setup should be able to given an api return starting state. 
        I think I need to clean up this method before this test makes sense,
        but for now I will just say what I expect the test to do. 
        """

        env = self.getTestEnv()

        expected_state = [[0.0  for _ in range(7)]]

        state = env.setup()

        assert expected_state == state 

    def testStepEnv(self) -> None: 
        """
        Step should return state, reward and done.
        Probably also need to clean up this method. 
        """
        env = self.getTestEnv()

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
        env = self.getTestEnv()

        actions = [[0.0, 0.0]]

        env.act(actions) # Currently this has no return val, will change.

class TestProductFeatures:

    pass
