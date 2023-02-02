import datetime
import config
from agent.Agent import Agent, MetricLogger
from environment.Env import Environment
from exchange_api.coinbase_api import CoinbaseAPI
import torch
from pathlib import Path
import time
import numpy as np

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
    agent = Agent(state_dim, action_dim, save_dir)

    checkpoint = None
    if checkpoint is not None:
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.target.load_state_dict(checkpoint["critic_target"])
        agent.exploration_rate = checkpoint["exploration_rate"]
        agent.actor_optim.load_state_dict(checkpoint["actor_optim"])
        agent.critic_optim.load_state_dict(checkpoint["critic_optim"])
        agent.actor_swa.load_state_dict(checkpoint["actor_swa"])
        agent.critic_swa.load_state_dict(checkpoint["critic_swa"])

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
                action = agent.act(state)

                env.act(action)
                time.sleep(0.5)
                try:
                    next_state, reward, done = env.step()
                except:
                    time.sleep(0.5)
                    next_state, reward, done = env.step()

                agent.cache(state, next_state, action, reward, done)

                actor_loss, critic_loss, alpha_loss = agent.learn()

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

                step = agent.curr_step

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
                    episode=e, epsilon=agent.exploration_rate, step=agent.curr_step
                )
