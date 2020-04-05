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
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from unityagents import UnityEnvironment

from agent import TennisMultiAgent
from maddpg import MADDPG


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":

    # Configuration
    n_episodes = 5000

    # Set seed
    seed_everything(42)

    # Unitiy environment
    env = UnityEnvironment("./Tennis_Linux/Tennis.x86_64")

    # Agent
    agent = TennisMultiAgent(state_size=24, action_size=2, n_agents=2)

    # DDPG
    maddpg = MADDPG(env=env, agent=agent)
    scores, avg_scores = maddpg.train(n_episodes=n_episodes)

    # Close the environment
    env.close()

    # Plot scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.linspace(1, n_episodes + 1, n_episodes), scores)
    ax.plot(np.linspace(1, n_episodes + 1, n_episodes), avg_scores)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Score per Episode")
    ax.set_title("Training progress of MADDPG model")

    fig.savefig('train_scores.png')
