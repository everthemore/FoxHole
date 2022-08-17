import numpy as np

# OpenAI gym
import gym
from gym import spaces

# Import fox-in-a-hole game
from unitary.examples.fox_in_a_hole.fox_in_a_hole import *

class FoxHole(gym.Env):
    def __init__(self, num_holes, max_steps):
        super(FoxHole, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.num_holes = num_holes
        self.max_steps = max_steps

        # Possible actions
        self.action_space = spaces.Discrete(num_holes)
        self.observation_space = spaces.Box(low=np.zeros(max_steps), high=np.ones(max_steps), dtype=np.uint8)
        self.reset()

    def step(self, action):
        self.history[self.s] = action
        self.s += 1

        # Perform action
        done = self.game.check_guess(action)
        # Fox makes a move
        self.game.take_random_move()
        # Update observation
        observation = np.array(self.history) #self.game.get_probabilities()

        # Perhaps we won?
        reward = 1 if done else 0

        if self.s >= self.max_steps:
            done = True
            reward = 0

        return observation, reward, done, {}

    def reset(self):
        # Create a new game
        self.game = QuantumGame(iswap=True, hole_nr=self.num_holes)
        self.game.initialize_state()
        self.s = 0

        self.history = np.zeros(self.max_steps) #self.game.get_probabilities()
        return np.array(self.history)

    def render(self, mode='human'):
        print("Game History:")
        for i in range(self.s):
            print("Move {0}: {1}".format(i,self.history[i]))

    def close (self):
        return

if __name__ == "__main__":
    # Test env
    env = FoxHole(5,10)

    env.reset()

    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render()
