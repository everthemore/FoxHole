import numpy as np
from FoxHoleEnvironment import FoxHole
from stable_baselines3 import PPO

env = FoxHole(3,5)
model = PPO.load("ppo_foxhole")

count = 0
num_games_to_play = 100
is_game_quantum = np.zeros(num_games_to_play)
is_game_won = np.zeros(num_games_to_play)

for g in range(num_games_to_play):
    print("\nPlaying game %d"%g)

    obs = env.reset()
    done = False

    classical = True
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        if info['move'] == 'quantum':
            is_game_quantum[g] = 1

    if( reward == 1 ):
        is_game_won[g] = 1
        print("Won!")
    else:
        print("Lost!")

print("Won {0} out of {1} games".format(np.count_nonzero(is_game_won), num_games_to_play))

quantum_game_indices = np.where(is_game_quantum == 1)
print("Won {0} out of {1} *quantum* games".format(np.count_nonzero(is_game_won[quantum_game_indices]), len(quantum_game_indices[0])))
