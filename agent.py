#  MIT License
#
#  Copyright (c) 2020 Peter Pesti <pestipeti@gmail.com>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import torch
import numpy as np
import random
import copy

import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from model import TennisActorModel, TennisCriticModel

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4

LEARN_AFTER_EVERY = 6
LEARN_ITER = 3

SIGMA_DECAY = 0.95
SIGMA_MIN = 0.005

FC1 = 400
FC2 = 300

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode(sa):
    """
    Encode an Environment state or action list of array, which contain multiple agents action/state information,
    by concatenating their information, thus removing (but not loosing) the agent dimension in the final output.

    The ouput is a list intended to be inserted into a buffer memmory originally not designed to handle multiple
    agents information, such as in the context of MADDPG)

    Params
    ======
            sa (list) : List of Environment states or actions array, corresponding to each agent

    """
    return np.array(sa).reshape(1, -1).squeeze()


def decode(size, num_agents, id_agent, sa, debug=False):
    """
    Decode a batch of Environment states or actions, which have been previously concatened to store
    multiple agent information into a buffer memmory originally not designed to handle multiple
    agents information(such as in the context of MADDPG)

    This returns a batch of Environment states or actions (torch.tensor) containing the data
    of only the agent specified.

    Params
    ======
            size (int): size of the action space of state spaec to decode
            num_agents (int) : Number of agent in the environment (and for which info hasbeen concatenetaded)
            id_agent (int): index of the agent whose informationis going to be retrieved
            sa (torch.tensor) : Batch of Environment states or actions, each concatenating the info of several
                                agents (This is sampled from the buffer memmory in the context of MADDPG)
            debug (boolean) : print debug information

    """

    list_indices = torch.tensor([idx for idx in range(id_agent * size, id_agent * size + size)]).to(device)
    out = sa.index_select(1, list_indices)

    if debug:
        print("\nDebug decode:\n size=", size, " num_agents=", num_agents, " id_agent=", id_agent, "\n")
        print("input:\n", sa, "\n output:\n", out, "\n\n\n")

    return out


class TennisAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents=1) -> None:
        """Initialize a TennisAgent.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            n_agents (int): Number of agents in the environment
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents

        # keeps track of how many steps have been taken.
        self.steps = 0

        # Actor network (w/ Target Network)
        self.actor_local = TennisActorModel(state_size, action_size, fc1_units=FC1, fc2_units=FC2).to(device)
        self.actor_target = TennisActorModel(state_size, action_size, fc1_units=FC1, fc2_units=FC2).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic network (w/ Target network)
        # Note : in MADDPG, critics have access to all agents obeservations and actions
        self.critic_local = TennisCriticModel(state_size * n_agents, action_size * n_agents,
                                              fc1_units=FC1, fc2_units=FC2).to(device)
        self.critic_target = TennisCriticModel(state_size * n_agents, action_size * n_agents,
                                               fc1_units=FC1, fc2_units=FC2).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # Noise
        self.noise = OUNoise(action_size)

        # Replay memory (We will use shared memory for all of the agents (multi-agent ddpg)
        self.memory = None

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        pass  # Not used in MADDPG

    def act(self, state, noise=0.0):
        """Returns actions for given state as per current policy.

        Args:
            state (np.ndarray): current state
            noise (float): Noise ratio (0.0 -> no noise)

        Returns:
            (np.ndarray): Actions (clipped -1 .. 1)
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()

        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        # add noise to actions
        action = action + (self.noise.sample() * noise)

        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma=0.99):
        """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

            Where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

        Args:
            experiences (Tuple[torch.tensor]): tuple of (s, a, r, s', next)
            gamma (float): discount factor
        """
        pass  # Not used in MADDPG

    @staticmethod
    def hard_copy_weights(target, source):
        """Copy weights from source to target network during initialization"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update_weights(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (nn.Module): PyTorch model; weights will be copied from)
            target_model (nn.Module): PyTorch model; weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()

    def save(self, filename):
        torch.save(self.actor_local.state_dict(), filename.format("_actor"))
        torch.save(self.critic_local.state_dict(), filename.format("_critic"))

    def load(self, filename):
        self.actor_local.load_state_dict(torch.load(filename.format("_actor")))
        self.critic_local.load_state_dict(torch.load(filename.format("_critic")))


class TennisMultiAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents):
        """Initialize a MADDPG Agent object.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            n_agents (int): Number of agents in the environment
        """

        super(TennisMultiAgent, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = n_agents

        # keeps track of how many steps have been taken.
        self.steps = 0

        # Instantiate Multiple TennisAgent
        self.agents = [TennisAgent(state_size, action_size, n_agents=n_agents) for i in range(n_agents)]

        # Instantiate Memory replay Buffer (shared between agents)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.steps += 1

        self.memory.add(encode(states), encode(actions), rewards, encode(next_states), dones)

        if (len(self.memory) > BATCH_SIZE) and (self.steps % LEARN_AFTER_EVERY == 0):

            assert (len(self.agents) == 2)  # Note: this code only expects 2 agents

            # Allow to learn several time in a row in the same episode
            for i in range(LEARN_ITER):
                # TODO: Generalize to n agents.

                # Update Agent #0
                self.maddpg_learn(self.memory.sample(), own_idx=0, other_idx=1)

                # Update Agent #1
                self.maddpg_learn(self.memory.sample(), own_idx=1, other_idx=0)

    def act(self, states, noise):
        """Return action to perform for each agents (per policy)"""
        return [agent.act(state, noise) for agent, state in zip(self.agents, states)]

    def maddpg_learn(self, experiences, own_idx, other_idx, gamma=GAMMA):
        """
        Update the policy of the MADDPG "own" agent. The actors have only access to agent own
        information, whereas the critics have access to all agents information.

        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

        where:
            actor_target(states) -> action
            critic_target(all_states, all_actions) -> Q-value

        TODO: Generalize to n agents.

        Args:
            experiences (Tuple[torch.tensor]): tuple of (s, a, r, s', next)
            own_idx (int) : index of the own agent to update in self.agents
            other_idx (int) : index of the other agent to update in self.agents
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Filter out the agent OWN states, actions and next_states batch
        own_states = decode(self.state_size, self.num_agents, own_idx, states)
        own_actions = decode(self.action_size, self.num_agents, own_idx, actions)
        own_next_states = decode(self.state_size, self.num_agents, own_idx, next_states)

        # Filter out the OTHER agent states, actions and next_states batch
        other_states = decode(self.state_size, self.num_agents, other_idx, states)
        other_actions = decode(self.action_size, self.num_agents, other_idx, actions)
        other_next_states = decode(self.state_size, self.num_agents, other_idx, next_states)

        # Concatenate both agent information (own agent first, other agent in second position)
        all_states = torch.cat((own_states, other_states), dim=1).to(device)
        all_actions = torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states = torch.cat((own_next_states, other_next_states), dim=1).to(device)

        agent = self.agents[own_idx]
        rewards = rewards[:, own_idx].unsqueeze(-1)
        dones = dones[:, own_idx].unsqueeze(-1)

        # ################
        # Update Critic

        # Get predicted next-state actions and Q values from target models
        all_next_actions = torch.cat((agent.actor_target(own_states),
                                      agent.actor_target(other_states)), dim=1).to(device)
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        agent.critic_optim.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optim.step()

        # ################
        # Update Actor
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                     dim=1).to(device)
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()

        # Minimize the loss
        agent.actor_optim.zero_grad()
        actor_loss.backward()
        agent.actor_optim.step()

        # ########################
        # Update target networks
        agent.soft_update_weights(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update_weights(agent.actor_local, agent.actor_target, TAU)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def save(self, filename):
        for i, agent in enumerate(self.agents):
            agent_i = filename.format("_" + str(i) + "{}")
            agent.save(agent_i)

    def load(self, filename):
        for i, agent in enumerate(self.agents):
            agent_i = filename.format("_" + str(i) + "{}")
            agent.load(agent_i)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, action_size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = action_size
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)  # use normal distribution
        self.state = x + dx
        return self.state


# From Udacity Deep Reinforcement Learning course.
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
