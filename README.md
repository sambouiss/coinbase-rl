# coinbase-rl
A reinforcement learning framework for market making on Coinbase. The framework is something I came up with to turn the market information into an RL enviroment. The part of the implementation I think is unique is using log returns instead of simple returns, and using market return as a stable baseline. E.g. Let $W_t$ denote wealth at time $t$ and $M_t$ denote the market portfolio at time $t$ given buy a half cash half stock position with continous rebalancing.

  $$ R_{\delta t} = log(W_{t+\delta t}) - log(W_t) -log(M_{t+\delta t}) + log(M_t)$$

This has a nice practical interpritability of being $\alpha$ in a finance context and also being a stable baseline in an RL context. 

Impirically taking this step sped up stability and convergence while training.


Currently WIP to do:
  - add tests for classes 
  - clean up code 


# Usage
To setup project run
```
pip install requirements.txt
```

Configure the coinbase api parameters for:
```
API_KEY = "your_key_here"
API_SECRET = "your_secret_key_here"
API_PASS = "your_api_password_here"
```

Then run the script.

```
python learn_mm.py
```

For parameters configurable by run time arguments simply add:

```
python learn_mm.py --actor_lr 0.01
```
