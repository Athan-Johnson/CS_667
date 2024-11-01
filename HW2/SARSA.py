import gymnasium as gym
import random
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Parsing
parser = argparse.ArgumentParser(description='Run SARSA on the icy lake.')
parser.add_argument('--iterations', type=int, default=10000, help='The number of iterations to train for, default is 10,000')
parser.add_argument('--show_final_policy', type=bool, default=False, help='Decide whether or not to show five games at the end for the user to watch, default is False')
args = parser.parse_args()



# Create the 4x4 Frozen Lake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Initialize parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 1 - (50 / args.iterations)
num_actions = env.action_space.n
num_states = env.observation_space.n
rewards = []
epoch = args.iterations / 10


# Initialize Q-table with zeros
policy = np.zeros((num_states, num_actions))

# this is the for loop we're going to be running the training in
for episode in tqdm.tqdm(range(args.iterations)):
	# Reset the environment to start a new episode
	state, info = env.reset()

	if random.uniform(0, 1) < epsilon:
		action = env.action_space.sample()
	else:
		action = np.argmax(policy[state, :])

	done = False
	while not done:

		# Step the environment with the chosen action
		next_state, reward, done, truncated, info = env.step(action)

		if random.uniform(0, 1) < epsilon:
			next_action = env.action_space.sample()
		else:
			next_action = np.argmax(policy[next_state, :])

		policy[state, action] += learning_rate * (reward + discount_factor * policy[next_state, next_action] - policy[state, action])

		state = next_state
		action = next_action

		# Print information about the current step
		# print(f"state: {state}, Reward: {reward}, Done: {done}, Info: {info}")
	
	# Decay epsilon to reduce exploration over time
	epsilon = max(min_epsilon, epsilon * epsilon_decay)

	if episode % epoch == 0:
		averageReward = 0
		for _ in range(10):
			state, info = env.reset()

			done = False
			while not done:
				# Implenent the policy
				action = np.argmax(policy[state, :])

				# Step the environment with the chosen action
				state, reward, done, truncated, info = env.step(action)
			
			averageReward += reward

		rewards.append(averageReward / 10)



# Close the environment when finished
env.close()

# Make the graph
plt.plot(range(len(rewards)),
		 rewards,
		 linestyle="-",
		 marker='.',
		 label="Rewards per epoch")
plt.grid()
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.savefig('Rewards_over_epochs_SARSA.png')
plt.show()

# run 100 times to get the win rate of the algorithm over 100 games
# Create the 4x4 Frozen Lake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

wins = 0
games = 10000
for episode in tqdm.tqdm(range(games)):
	# Reset the environment to start a new episode
	state, info = env.reset()

	done = False
	while not done:
		# Implenent the policy
		action = np.argmax(policy[state, :])

		# Step the environment with the chosen action
		state, reward, done, truncated, info = env.step(action)

		# Print information about the current step
		# print(f"state: {state}, Reward: {reward}, Done: {done}, Info: {info}")
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

	for episode in range(5):
		# Reset the environment to start a new episode
		state, info = env.reset()

		done = False
		while not done:
			# Render the current state of the environment
			env.render()

			# Implement the policy
			action = np.argmax(policy[state, :])

			# Step the environment with the chosen action
			state, reward, done, truncated, info = env.step(action)

			# Print information about the current step
			# print(f"state: {state}, Reward: {reward}, Done: {done}, Info: {info}")

	# Close the environment when finished
	env.close()
