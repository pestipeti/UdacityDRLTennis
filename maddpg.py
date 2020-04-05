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
import numpy as np

from collections import deque

from agent import TennisMultiAgent


class MADDPG:

    def __init__(self, env, agent: TennisMultiAgent) -> None:
        self.env = env
        self.agent = agent  # type: TennisMultiAgent

        # get the default brain
        self.brain_name = env.brain_names[0]

    def train(self, n_episodes=10000, n_steps=1000):

        all_scores = []
        all_avg_score = []
        all_steps = 0
        scores_last = deque(maxlen=100)
        solved = False
        noise = 0.0

        for i_episode in range(1, n_episodes + 1):
            # Reset environment, state and scores
            self.agent.reset()
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations

            # We will tracking scores for each agent
            episode_scores = np.zeros(self.agent.num_agents)

            while True:
                all_steps += 1

                # A single step of interaction with the environment for each agent
                actions = self.agent.act(states, noise=noise)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                self.agent.step(states, actions, rewards, next_states, dones)

                # Sum up rewards separately for each agent
                episode_scores += np.array(rewards)

                # Prepare for next timestep of iteraction
                # new states become the current states
                states = next_states

                # Check if any of the agents has finished. Finish to keep all
                # trajectories in this batch the same size.
                if np.any(dones):
                    break

            # Update scores
            episode_score = np.max(episode_scores)
            scores_last.append(episode_score)
            all_scores.append(episode_score)
            all_avg_score.append(np.mean(scores_last))

            print('\rEpisode {}\tAverage Score: {:.4f}\tEpisode score'
                  ' (max over agents): {:.4f}'.format(i_episode, np.mean(scores_last), episode_score), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.4f}\tEpisode score'
                      ' (max over agents): {:.4f}'.format(i_episode, np.mean(scores_last), episode_score))

            if np.mean(scores_last) >= 0.5 and not solved:
                print("\nEnvironment solved in {} episodes. Average score over the "
                      "last 100 episodes: {:.4f}".format(i_episode, np.mean(scores_last)))
                solved = True

            self.agent.save("checkpoint{}.pth")

        return all_scores, all_avg_score

    def test(self, n_episodes=10):

        scores = []

        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            states = env_info.vector_observations
            self.agent.reset()

            # We will tracking scores for each agent
            episode_scores = np.zeros(self.agent.num_agents)

            while True:
                actions = self.agent.act(states, noise=0.0)
                env_info = self.env.step(actions)[self.brain_name]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                # Sum up rewards separately for each agent
                episode_scores += np.array(rewards)

                # Prepare for next timestep of iteraction
                # new states become the current states
                states = next_states

                # Check if any of the agents has finished. Finish to keep all
                # trajectories in this batch the same size.
                if np.any(dones):
                    break

            scores.append(np.max(episode_scores))

        return scores
