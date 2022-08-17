import numpy as np
from FoxHoleEnvironment import FoxHole
from stable_baselines3 import PPO

env = FoxHole(5,10)
model = PPO.load("ppo_foxhole")

count = 0
trajectories = []
for g in range(10):
    print("\nPlaying game %d"%g)
    obs = env.reset()
    done = False

    trajectory = []
    while not done:
        print(obs)
        action, _states = model.predict(obs)
        trajectory.append(action)
        print("Action: ", action)
        obs, rewards, done, info = env.step(action)
        env.render()

        if( rewards == 1 ):
            count += 1
            print("Won!")

    trajectories.append(trajectory)
print("Won %d out of 10 games"%count)

# import matplotlib.pyplot as plt
# for t in trajectories:
#     plt.plot(t)
# plt.show()
