import datetime
from typing import Optional
import config
from agent import Agent, MetricLogger
from environment.env import Environment
from exchange_api.coinbase_api import CoinbaseAPI
import torch
from pathlib import Path
import time
import numpy as np
import sys
import argparse
import agent.config

parser = argparse.ArgumentParser(description="coinbase reinforcement learning framework")

parser.add_argument(
    "--actor_lr", 
    type = float,  
    default = agent.config.actor_lr,
    help= "The learning rate for the actor optimizer."
    )
parser.add_argument(
    "--actor_swa_lr",
    type = float,
    default = agent.config.actor_swa_lr,
    help= "The Learning rate for the actor optimizer SWA."
    )
parser.add_argument(
    "--actor_swa_start",
    type = int,
    default = agent.config.actor_swa_start,
    help= "Number of training steps before we apply SWA to actor."
    )
parser.add_argument(
    "--actor_swa_freq",
    type = int,
    default = agent.config.actor_swa_freq,
    help= "Number of training steps between applying SWA to actor."
    )

parser.add_argument(
    "--critic_lr", 
    type = float,  
    default= agent.config.critic_lr, 
    help= "The learning rate for the critic optimizer."
    )
parser.add_argument(
    "--critic_swa_lr", 
    type = float,  
    default= agent.config.critic_swa_lr,
    help = "The learning rate for the critic optimizer SWA."
    )
parser.add_argument(
    "--critic_swa_start",
    type = int,
    default = agent.config.critic_swa_start,
    help= "Number of training steps before we apply SWA to critic."
    )
parser.add_argument(
    "--critic_swa_freq",
    type = int,
    default = agent.config.critic_swa_freq,
    help= "Number of training steps between applying SWA to critic."
    )

parser.add_argument(
    "--hidden_size", 
    type = float,  
    default= agent.config.hidden_size,
    help = "The widith of the hidden dimensions."
    )
parser.add_argument("--check_point", 
                    type = Optional[str],  
                    default = agent.config.check_point,
                    help="Optional path to checkpoint default is none.")

loss_actor = []
loss_critic = []
loss_alpha = []

if config.USE_SANDBOX:
    api_url = config.api_url_sb
    API_KEY = config.API_KEY_SB
    API_SECRET = config.API_SECRET_SB
    API_PASS = config.API_PASS_SB
else:
    api_url = config.api_url
    API_KEY = config.API_KEY
    API_SECRET = config.API_SECRET
    API_PASS = config.API_PASS

if __name__ == "__main__":
    api = CoinbaseAPI(api_url, API_KEY, API_SECRET, API_PASS)
    session_length = 60 * 60
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=session_length)
    products = ["AMP-USD"]
    env = Environment(
        start_time, end_time, api, products, session_length=session_length
    )
    state_dim = (len(products), 7)
    action_dim = (len(products), 2)
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)
    rl_agent = Agent(state_dim, action_dim, save_dir)

    checkpoint = None
    if checkpoint is not None:
        rl_agent.actor.load_state_dict(checkpoint["actor"])
        rl_agent.critic.load_state_dict(checkpoint["critic"])
        rl_agent.target.load_state_dict(checkpoint["critic_target"])
        rl_agent.exploration_rate = checkpoint["exploration_rate"]
        rl_agent.actor_optim.load_state_dict(checkpoint["actor_optim"])
        rl_agent.critic_optim.load_state_dict(checkpoint["critic_optim"])
        rl_agent.actor_swa.load_state_dict(checkpoint["actor_swa"])
        rl_agent.critic_swa.load_state_dict(checkpoint["critic_swa"])

    logger = MetricLogger(save_dir)

    episodes = 100
    for e in range(episodes):
        rewards_sum = 0
        try:
            while api.cancelAllOrders():
                pass

            try:
                state = env.setup()
            except:
                state = env.setup()

            while True:
                print(state)
                action = rl_agent.act(state)

                env.act(action)
                time.sleep(0.5)
                try:
                    next_state, reward, done = env.step()
                except:
                    time.sleep(0.5)
                    next_state, reward, done = env.step()

                rl_agent.cache(state, next_state, action, reward, done)

                actor_loss, critic_loss, alpha_loss = rl_agent.learn()

                rewards_sum += np.mean(reward)
                logger.log_step(np.mean(reward), actor_loss, critic_loss)

                if actor_loss:
                    loss_actor.append(actor_loss)
                if critic_loss:
                    loss_critic.append(critic_loss)
                if alpha_loss:
                    loss_alpha.append(alpha_loss)
                state = next_state

                if done[0]:
                    break

                step = rl_agent.curr_step

                if step % 20 == 0:

                    print(
                        "cumulative reward: {} pos pnl: {} mean loss actor: {}  mean loss critic: {}".format(
                            rewards_sum,
                            env.pos_pnl,
                            np.mean(loss_actor),
                            np.mean(loss_critic),
                            np.mean(loss_alpha),
                        )
                    )

        except Exception as ex:
            print(ex)
        finally:

            logger.log_episode()
            if e % 3 == 0:
                logger.record(
                    episode=e, epsilon=rl_agent.exploration_rate, step=rl_agent.curr_step
                )
