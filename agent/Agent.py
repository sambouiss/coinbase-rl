import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
from collections import namedtuple
import time, datetime
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torchcontrib.optim import SWA

Action = namedtuple('Action',['bid_offset','bid_prop','ask_offset','ask_prop'])

class Agent:
    def __init__(self,state_dim,action_dim,save_dir):

        self.state_dim=state_dim
        self.action_dim =action_dim
        self.save_dir = save_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor = DeterministicActorNet(state_dim,action_dim,hidden_layers=256).float()
        self.critic = CriticNet(state_dim,action_dim,hidden_layers=256).float()
        self.actor = self.actor.to(device=self.device)
        self.critic = self.critic.to(device=self.device)
        self.target = copy.deepcopy(self.critic)
        for p in self.target.parameters():
            p.requires_grad = False
        
        self.exploration_rate = 0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0
        self.curr_step = 0

        self.save_every = 1024
        self.gamma = 0.9

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.00005)
        self.actor_swa = SWA( self.actor_optim, swa_start=10, swa_freq=5, swa_lr=0.00005)
        self.actor_swa.defaults = self.actor_swa.optimizer.defaults
        self.actor_swa.param_groups = self.actor_swa.optimizer.param_groups
        self.actor_swa.state = self.actor_swa.optimizer.state
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.00005)
        self.critic_swa = SWA( self.actor_optim, swa_start=10, swa_freq=5, swa_lr=0.00005)
        self.critic_swa.defaults = self.critic_swa.optimizer.defaults
        self.critic_swa.param_groups = self.critic_swa.optimizer.param_groups
        self.critic_swa.state = self.critic_swa.optimizer.state
        self.loss_fn = torch.nn.MSELoss() 
        self.max_grad_norm=0.5
        self.num_epochs = 8
        self.burnin = 128  # min. experiences before training
        self.learn_every = 128  # no. of experiences between updates to Q_online
        self.sync_every = 1024  # no. of experiences between Q_target & Q_online sync
        self.memory = deque(maxlen=12000)
        self.batch_size = 128
        self.alpha = 0
        self.target_entropy = -torch.prod(torch.Tensor(self.action_dim[1]).to(self.device)).float().item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device).float()
        self.alpha_optim =  torch.optim.Adam([self.log_alpha], lr=0.00005)
        self.tau = .3

    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(LazyFrame): A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action Mario will perform
    """
        # # EXPLORE
        # if np.random.rand() < self.exploration_rate:
        #     action = np.random.normal(0,10,self.action_dim)

        # # EXPLOIT
        # else:
            #state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        #print(self.actor(state).sample().detach().reshape(-1,self.action_dim[1]).data.numpy())
        action,_,_= self.actor.sample(state)

        action = action.detach().reshape(-1,self.action_dim[1]).cpu().data.numpy()

            

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        # def first_if_tuple(x):
        #     return x[0] if isinstance(x, tuple) else x
        # state = first_if_tuple(state).__array__()
        # next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state, device=self.device)
        next_state = torch.tensor(next_state, device=self.device)
        action = torch.tensor([action], device=self.device)
        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        self.memory.append((state, next_state, action, reward, done,))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    @torch.no_grad()
    def td_target(self, reward, actions, next_state, next_state_log_pi, done):
        q1_next, q2_next = self.target(next_state, actions)
        min_qf_next_taget = torch.min(q1_next, q2_next)-self.alpha*next_state_log_pi
        return (reward+(1-done.float())*self.gamma*min_qf_next_taget).float()
    
    def td_estimate(self, state, action):
        q1, q2 = self.critic(state, action)
        return q1, q2

    def save(self):
        save_path = (
            self.save_dir / f"agent_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(actor=self.actor.state_dict(),
                critic=self.critic.state_dict(), 
                exploration_rate=self.exploration_rate,
                critic_target=self.target.state_dict(),
                critic_optim=self.critic_optim.state_dict(),
                actor_optim=self.actor_optim.state_dict(),
                critic_swa=self.critic_swa.state_dict(),
                actor_swa=self.actor_swa.state_dict()
                ),
            save_path,
        )
        print(f"Agent saved to {save_path} at step {self.curr_step}")

    def updated_critic(self, td_estimate, qf1, qf2):
        loss_1 = self.loss_fn(td_estimate, qf1)
        loss_2 = self.loss_fn(td_estimate,qf2)
        loss = loss_1+loss_2
        self.critic_swa.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in self.critic_swa.param_groups for p in g["params"]], self.max_grad_norm) # gradient clipping
        self.critic_swa.step()
        return loss.item()

    @torch.no_grad()
    def get_advantage(self, values, td_target):
        return td_target-values

    def update_actor(self, log_probs, min_qf_pi):
        
        loss =(-min_qf_pi).mean()
        self.actor_swa.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in self.actor_swa.param_groups for p in g["params"]], self.max_grad_norm)
        self.actor_swa.step()
        return loss.item()
    
    def update_alpha(self, log_probs):
        loss = -(self.log_alpha*(log_probs+self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([p for g in self.alpha_optim.param_groups for p in g["params"]], self.max_grad_norm)
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        return loss.item()

    def sync_Q_target(self):
        for target_param,param in zip(self.target.parameters(),self.critic.parameters()):
            target_param.data.copy_(target_param.data*(1.0-self.tau)+param.data*self.tau)

    def learn(self):
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
           

        if self.curr_step < self.burnin:
            return None, None, None

        if self.curr_step % self.learn_every != 0:
            return None, None, None

        for _ in range(self.num_epochs):
            state, next_state, action, reward, done = self.recall()
            reward = reward.reshape(self.batch_size,self.state_dim[0],1)
            done = done.reshape(self.batch_size,self.state_dim[0],1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            
           
            td_target = self.td_target(reward, next_state_action, next_state, next_state_log_pi, done)
            #print(tuple(item.shape for item in (reward, next_state_action, next_state, next_state_log_pi, done)))
            qf1, qf2 = self.critic(state, action)
           
            critic_loss = self.updated_critic(td_target, qf1, qf2)
            

            

            pi, log_pi, _  = self.actor.sample(state)
            qf1_pi, qf2_pi = self.critic(state, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = self.update_actor(log_pi, min_qf_pi)

            alpha_loss = 0

        self.actor_swa.swap_swa_sgd()
        self.critic_swa.swap_swa_sgd()  
        #print(actor_loss, critic_loss, alpha_loss)
        return (actor_loss, critic_loss, alpha_loss)
    
class ActorNet(nn.Module):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    epsilon = 1e-6
    def __init__(self,state_dim,action_dim,activation=nn.LeakyReLU,hidden_layers=64):
        super().__init__()
        n_1,features = state_dim
        n_2,num_actions = action_dim

        assert n_1 == n_2

        self.shared = nn.Sequential(
            nn.Linear(features, hidden_layers),
            activation(),
            nn.Linear(hidden_layers, hidden_layers),
            activation(),
            nn.Linear(hidden_layers,hidden_layers)
        )

        self.mean_linear = nn.Linear(hidden_layers,num_actions)
        self.log_std_linear = nn.Linear(hidden_layers,num_actions)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, X):
        X = self.shared(X)
        mean = self.mean_linear(X)
        log_std = self.log_std_linear(X)
        log_std = torch.clamp(log_std,min = ActorNet.LOG_SIG_MIN, max=ActorNet.LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):

        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + ActorNet.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class CriticNet(nn.Module):
    
    def __init__(self, state_dim, action_dim, activation=nn.LeakyReLU,hidden_layers=64):
        super().__init__()
        _, features = state_dim
        _, actions = action_dim 
        self.features = features
        self.actions = actions
        self.q_1 = nn.Sequential(
            nn.Linear(features+actions, hidden_layers),
            activation(),
            nn.Linear(hidden_layers, hidden_layers),
            activation(),
            nn.Linear(hidden_layers, 1),
        )

        self.q_2 = nn.Sequential(
            nn.Linear(features+actions, hidden_layers),
            activation(),
            nn.Linear(hidden_layers, hidden_layers),
            activation(),
            nn.Linear(hidden_layers, 1),
        )

    def forward(self, state, action):
        
        X = torch.cat([state.reshape(-1, self.features).float(), action.reshape(-1,self.actions).float()], 1)
        return self.q_1(X).unsqueeze(1), self.q_2(X).unsqueeze(1)


class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


class DeterministicActorNet(nn.Module):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    epsilon = 1e-6
    def __init__(self,state_dim,action_dim,activation=nn.LeakyReLU,hidden_layers=64):
        super().__init__()
        n_1,features = state_dim
        n_2,num_actions = action_dim

        assert n_1 == n_2
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.shared = nn.Sequential(
            nn.Linear(features, hidden_layers),
            activation(),
            nn.Linear(hidden_layers, hidden_layers),
            activation(),
            nn.Linear(hidden_layers,hidden_layers),
            activation(),
            nn.Linear(hidden_layers,num_actions)
        )

        
        self.noise =torch.Tensor(num_actions).to(self.device)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, X):
        mean = self.shared(X)
        return mean

    def sample(self, state):

        mean = self.forward(state)
        noise = self.noise.normal_(0.,std=.1)
        noise.clamp(-.25,.25)
        action = mean+noise
        return action, torch.zeros(action.shape[0],action.shape[1],device=self.device).unsqueeze(1), mean