import numpy as np
import os

from FoxHoleEnvironment import FoxHole
from stable_baselines3 import PPO

env = FoxHole(3,5)

os.makedirs("./logs/", exist_ok=True)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

print("Training")
model.learn(total_timesteps=1000)#, reset_num_timesteps=False)

model.save("ppo_foxhole")
