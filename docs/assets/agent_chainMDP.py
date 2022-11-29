from torch import nn
from torch import optim
import torch
from collections import deque
import numpy as np

# Implement bootstrapped dqn agent


class agent():
    def __init__(self,
                 nAction=2,
                 K=10,
                 feature_size=10,
                 n_samples=10,
                 hidden_size=16,
                 buffer_size=1000,
                 mask_distribution="poisson",
                 lr=1e-4,
                 reset_tol=1,
                 reward_update_rate=1,
                 reset_count_threshold=5,
                 discount=1,
                 pause_sample=0,
                 exploit_prob_step=1 / 300,
                 exploit_prob_min=0.1,
                 exploit_prob_max=0.9,
                 exploit_rwd_maxlen=5,
                 ):
        self.nAction = nAction
        self.K = K
        self.feature_size = feature_size
        self.feature_size_check = False
        self.n_samples = n_samples
        self.mask_distribution = mask_distribution
        self.reset_tol = reset_tol
        self.reward_update_rate = reward_update_rate
        self.reset_count_threshold = reset_count_threshold
        self.discount = discount
        self.buffer_size = buffer_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.action_k = 0
        self.pause_sample = pause_sample
        self.exploit_action = False
        self.exploit_prob = exploit_prob_min
        self.exploit_prob_step = exploit_prob_step
        self.exploit_rwd_buffer = deque(maxlen=exploit_rwd_maxlen)
        self.explore_max_rwd = 0
        self.exploit_prob_min = exploit_prob_min
        self.exploit_prob_max = exploit_prob_max

        self.exploit_q = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.nAction)
        )
        self.exploit_q_optimizer = optim.Adam(self.exploit_q.parameters(),
                                              lr=lr)

        self.explore_q_list = [nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.nAction)
        )] * K
        self.explore_q_optimzer_list = [optim.Adam(self.explore_q_list[k].parameters(), lr=lr)
                                        for k in range(K)]

        self.buffer = deque(maxlen=buffer_size)
        self.avg_reward_list = np.zeros(self.K, dtype=np.float32)
        self.reward_list = []
        self.reset_count = [0] * K
        self.wait_until_sample = np.zeros(self.K, dtype=np.int64)

    def update_buffer(self, s, action, reward, ns, done, mask):
        s = self.np_to_torch(s)
        ns = self.np_to_torch(ns)
        self.buffer.append((s, action, reward, ns, done, mask))

    def mask_fn(self):
        if self.mask_distribution == "poisson":
            return np.random.poisson(lam=1, size=self.K)
        elif self.mask_distribution == "uniform":
            return np.random.uniform(size=self.K)
        else:
            raise NotImplementedError

    def load_weights(self):
        # load weights from q_k.pth
        for k in range(self.K):
            try:
                self.explore_q_list[k].load_state_dict(
                    torch.load(f"task1_q_{k}.pth"))
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"task1_q_{k}.pth not found.")

    def action(self, state):
        state = self.np_to_torch(state)

        # If necessary, reset DQN to match the shape of provided state features.
        if not self.feature_size_check:
            if self.feature_size != len(state):
                self.feature_size = len(state)
                self.explore_q_list = [nn.Sequential(
                    nn.Linear(self.feature_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.nAction)
                )] * self.K
                self.explore_q_optimzer_list = [
                    optim.Adam(self.explore_q_list[k].parameters(),
                               lr=self.lr) for k in range(self.K)]

                self.exploit_q = nn.Sequential(
                    nn.Linear(self.feature_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.nAction)
                )
                self.exploit_q_optimizer = optim.Adam(
                    self.exploit_q.parameters(), lr=self.lr)

            self.feature_size_check = True

        if self.exploit_action:
            action = torch.argmax(self.exploit_q(state))
        else:
            if 0 <= self.action_k < self.K:
                action = torch.argmax(self.explore_q_list[self.action_k](state))
            else:
                raise ValueError("k must be integer in [0, K)")
        return action.item()

    @staticmethod
    def reset_layer(layer: nn.Module):
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()

    @staticmethod
    def np_to_torch(x: np.ndarray):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        else:
            return x

    def reset_q(self, k: int):
        self.explore_q_list[k].apply(self.reset_layer)

    def step(self, k, loss):
        # Backpropagate the loss
        if isinstance(k, int):
            self.explore_q_optimzer_list[k].zero_grad()
        else:
            self.exploit_q_optimizer.zero_grad()
        loss.backward()
        if isinstance(k, int):
            self.explore_q_optimzer_list[k].step()
        else:
            self.exploit_q_optimizer.step()

    def update_rwd_info(self, rwd):
        if self.exploit_action:
            self.exploit_rwd_buffer.append(rwd)
        else:
            self.explore_max_rwd = max(self.explore_max_rwd, rwd)

        if min(self.exploit_rwd_buffer) >= self.explore_max_rwd:
            self.exploit_prob = min(self.exploit_prob_max,
                                    self.exploit_prob + self.exploit_prob_step)
        else:
            self.exploit_prob = max(self.exploit_prob_min,
                                    self.exploit_prob - self.exploit_prob_step)

    def reset_stall(self, epi, rwd, verbose=False):
        # Observe (reward) - (EMA of reward) of the chosen DQN agent and increment count if it is small
        # Reset the count if such change is large enough
        rwd_change = rwd - self.avg_reward_list[self.action_k]

        if abs(rwd_change) < self.reset_tol:
            self.reset_count[self.action_k] += 1
        else:
            self.reset_count[self.action_k] = 0

        # If reset_count is large enough, reset the DQN agent
        if self.reset_count[self.action_k] >= self.reset_count_threshold:
            if verbose:
                print(f"Reset Q_{self.action_k}")
            self.reset_q(self.action_k)
            self.reset_count[self.action_k] = 0
            self.wait_until_sample[self.action_k] = self.pause_sample

        # Save EMA(exponential moving average) of reward for the chosen agent for each episode
        self.avg_reward_list[self.action_k] += self.reward_update_rate * rwd_change
        self.reward_list.append((epi, self.action_k, rwd))

    def update(self, replays, k):
        loss = 0
        assert isinstance(k, int) or k == "exploit"

        if isinstance(k, int):
            for s, a, r, ns, d, mask in replays:
                next_q = self.explore_q_list[k](ns)
                next_a = torch.argmax(next_q)
                # Q-learning update
                target = r + (1 - d) * self.discount * next_q[next_a]
                loss += (target - self.explore_q_list[k](s)[a])**2

            loss *= mask[k]
            self.step(k, loss)

        else:
            for s, a, r, ns, d, _ in replays:
                next_q = self.exploit_q(ns)
                next_a = torch.argmax(next_q)
                # Q-learning update
                target = r + (1 - d) * self.discount * next_q[next_a]
                loss += (target - self.exploit_q(s)[a])**2
            self.step(k, loss)

        return loss.item()

    def choose_action_k(self):
        # Choose action_k to determine action w.r.t Q_{action_k}
        avg_rwd = np.array(self.avg_reward_list, copy=True)
        assert (self.wait_until_sample >= 0).all()
        avg_rwd[self.wait_until_sample > 0] = -np.inf

        # Choose action_k based on EMA of reward
        # Larger probability for larger EMA
        # Uniform probability if all EMA's are same for all DQN agents

        # set p as a softmax value of avg_rwd
        p = np.exp(avg_rwd) / np.sum(np.exp(avg_rwd))
        action_k = np.random.choice(np.arange(self.K), p=p)

        self.wait_until_sample = np.where(self.wait_until_sample > 0,
                                          self.wait_until_sample - 1,
                                          0)

        return action_k

    def reset_agent(self):
        # Reset all agents
        for k in range(self.K):
            self.reset_q(k)
        self.buffer = deque(maxlen=self.buffer_size)
        self.avg_reward_list = np.zeros(self.K, dtype=np.float32)
        self.reward_list = []
        self.reset_count = [0] * self.K

    def save_weights(self):
        for k in range(self.K):
            torch.save(self.explore_q_list[k].state_dict(), f"q_{k}.pth")
