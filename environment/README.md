# Environment 

This module contains: 
    - The environment which manages the state, action, and reward.
    - The product feautres which manage product level information e.g. producing states and using actions.

The episode length is set to one hour. I found that one hour was a good length to get a couple different market conditions to test with each strategy. The episodes consist of .5 second time increments because this should improve the markov-ness of the state transition dynamics.

The state space is set up such that each feature is a normalized relative measure over the session length. The idea behind this is that by making the state space in this way we traverse a reasonably bounded region so that states should be relatively similar across episodes. Theoretically this should also imporve convergence because it truncates the problem to an enclosed region instead of being unbounded. 

# Tests

To run the test run 
```
python -m pytest -q env_test.py
```
