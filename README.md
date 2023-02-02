# coinbase-rl
A reinforcement learning framework for market making on Coinbase. The framework is something I came up with to turn the market information into an RL enviroment.

The Agent is implemented using D3PG and SWA. 

Currently WIP to do:
  - add tests for classes 
  - clean up code 
  - create args for running 
  - add config files for parameters for model

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

Adding command line arguments and also more configure options is on my to do list.
