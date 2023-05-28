import gym
import torch
import copy
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import cv2
from argparse import ArgumentParser

GAME_SETTINGS = {
    "cartpole": {
        "in_num": 4,
        "out_num": 2,
        "environment_name": "CartPole-v1",
    },
    "acrobot": {
        "in_num": 6,
        "out_num": 3,
        "environment_name": "Acrobot-v1",
    },
    "pendulum": {
        "in_num": 3,
        "out_num": 1,
        "environment_name": "Pendulum-v1",
    },
    "mountain_car": {
        "in_num": 2,
        "out_num": 3,
        "environment_name": "MountainCar-v0",
    },
}


class Net(nn.Module):
    def __init__(self, in_num, out_num):
        super(Net, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(in_num, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, out_num),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(in_num, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def pick_action(self, observation, collector):
        observation = torch.from_numpy(observation).float()
        probabilities = self.actor(observation)
        distribution = Categorical(probabilities)
        action = distribution.sample()

        collector.states.append(observation)
        collector.actions.append(action)
        collector.action_logarithms.append(distribution.log_prob(action))
        return action.item()

    def evaluate(self, observation, action):
        probabilities = self.actor(observation)
        distribution = Categorical(probabilities)
        logarithm_probabilities = distribution.log_prob(action)
        entropy = distribution.entropy()
        Qvalue = self.critic(observation)
        return logarithm_probabilities, torch.squeeze(Qvalue), entropy


class DataCollector:
    def __init__(self, net, out_num, environment_name, gamma):
        self.net = net
        self.out_num = out_num
        self.env = gym.make(environment_name, render_mode="rgb_array")
        self.gamma = gamma
        self.rewards = []
        self.action_logarithms = []
        self.states = []
        self.render = False
        self.actions = []
        self.Qval = 0

    def clear_previous_batch_data(self):
        self.np_Qvals = []
        self.rewards = []
        self.action_logarithms = []
        self.states = []
        self.actions = []
        self.Qval = 0

    def calculate_qvals(self):
        Qval = 0
        Qvals = []
        for reward in reversed(self.rewards):
            Qval = reward + self.gamma * Qval
            Qvals.insert(0, Qval)
        return torch.tensor(Qvals)

    def collect_data_for(self, batch_size):
        current_state = self.env.reset()[0]
        print(np.array(current_state[0], dtype=np.float32))
        for simulation_step in range(batch_size):
            if self.render:
                img = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                cv2.imshow("test", img)
                cv2.waitKey(50)

            action = self.net.pick_action(current_state, self)
            observation, reward, done, _, _ = self.env.step(action)
            self.rewards.append(reward)
            current_state = observation
            if done or simulation_step == batch_size - 1:
                self.Qval = self.calculate_qvals()
                break

    def stack_data(self):
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.action_logarithms = torch.stack(self.action_logarithms)


class A2CTrainer:
    def __init__(
        self,
        in_num,
        out_num,
        environment_name,
        batch_size,
        gamma,
        beta_entropy,
        learning_rate,
        clip_size,
    ):
        self.net = Net(in_num, out_num)
        self.new_net = copy.deepcopy(self.net)
        self.batch_size = batch_size
        self.beta_entropy = beta_entropy
        self.learning_rate = learning_rate
        self.clip_size = clip_size
        self.optimizer = torch.optim.Adam(
            self.new_net.parameters(), lr=self.learning_rate
        )
        self.data = DataCollector(self.net, out_num, environment_name, gamma)

    def calculate_actor_loss(self, ratio, advantage):
        opt1 = ratio * advantage
        opt2 = torch.clamp(ratio, 1 - self.clip_size, 1 + self.clip_size) * advantage
        return (-torch.min(opt1, opt2)).mean()

    def calculate_critic_loss(self, advantage):
        return 0.5 * advantage.pow(2).mean()

    def train(self):
        self.data.clear_previous_batch_data()
        self.data.collect_data_for(batch_size=self.batch_size)
        self.data.stack_data()

        action_logarithms, Qval, entropy = self.new_net.evaluate(
            self.data.states, self.data.actions
        )

        ratio = torch.exp(action_logarithms - self.data.action_logarithms.detach())
        advantage = self.data.Qval - Qval.detach()
        actor_loss = self.calculate_actor_loss(ratio, advantage)
        critic_loss = self.calculate_critic_loss(advantage)

        loss = actor_loss + critic_loss + self.beta_entropy * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.net.load_state_dict(self.new_net.state_dict())
        return sum(self.data.rewards), self.net


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--game",
        choices=set(GAME_SETTINGS.keys()),
        default="cartpole",
        help="Game choice. Carpole by default.",
    )
    parser.add_argument("--render-interval", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=1000)
    ARGS = parser.parse_args()

    trainer = A2CTrainer(
        **GAME_SETTINGS[ARGS.game],
        batch_size=500,
        gamma=0.99,
        beta_entropy=0.001,
        learning_rate=0.001,
        clip_size=0.2,
    )
    best_result = 0
    in_which_episode = 0
    for episode in range(ARGS.episodes):
        if episode % ARGS.render_interval == 0:
            trainer.data.render = True
        else:
            trainer.data.render = False

        curr_result, _ = trainer.train()

        if curr_result > best_result:
            best_result = curr_result
            in_which_episode = episode
        print(
            f"{episode}. {curr_result}\tBest: {best_result} in episode {in_which_episode}"
        )
