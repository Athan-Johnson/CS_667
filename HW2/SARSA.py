import gymnasium as gym
import random
import tqdm
import argparse


# Parsing
parser = argparse.ArgumentParser(description='Run SARSA on the icy lake.')
parser.add_argument('--iterations', default=1000000, help='The number of iterations to train for, default is 1,000,000')
parser.add_argument('--show_final_policy', default=False, help='Decide whether or not to show five games at the end for the user to watch, default is False')
args = parser.parse_args()



# Create the 4x4 Frozen Lake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)


# this is the for loop we're going to be running the training in
for i in tqdm.tqdm(range(int(args.iterations))):
	# Reset the environment to start a new episode
	# the format here for our dict "policy" is obs, or the number assigned
	# to the state we're in, and the move from that state
	obs, info = env.reset()
	prevObs = obs

	done = False
	while not done:

		action = env.action_space.sample()

		# Step the environment with the chosen action
		obs, reward, done, truncated, info = env.step(action)
		# Print information about the current step
		# print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

# Close the environment when finished
env.close()


# run 100 times to get the win rate of the algorithm over 100 games
# Create the 4x4 Frozen Lake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

wins = 0
games = 10000
for i in tqdm.tqdm(range(games)):
	# Reset the environment to start a new episode
	obs, info = env.reset()

	done = False
	while not done:
		# Implenent the policy
		action = env.action_space.sample()

		# Step the environment with the chosen action
		obs, reward, done, truncated, info = env.step(action)

		# Print information about the current step
		# print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
		if reward == 1:
			wins += 1

print("Win rate: " + str((wins / games) * 100) + "%")

# Close the environment when finished
env.close()

# Run a final iteration of our trained algorithm with render_mode on so
# we can see how well it performs

if args.show_final_policy:
	# Create the 4x4 Frozen Lake environment
	env = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4", is_slippery=True)

	for i in range(5):
		# Reset the environment to start a new episode
		obs, info = env.reset()

		done = False
		while not done:
			# Render the current state of the environment
			env.render()

			# Implement the policy
			action = env.action_space.sample()

			# Step the environment with the chosen action
			obs, reward, done, truncated, info = env.step(action)

			# Print information about the current step
			# print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

	# Close the environment when finished
	env.close()