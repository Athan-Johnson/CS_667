import gymnasium as gym
import random
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


# Parsing
parser = argparse.ArgumentParser(description='Run SARSA on the icy lake.')
parser.add_argument('--iterations', type=int, default=10000, help='The number of iterations to train for, default is 10,000')
parser.add_argument('--show_final_policy', type=bool, default=False, help='Decide whether or not to show five games at the end for the user to watch, default is False')
parser.add_argument('--is_slippery', type=bool, default=True, help='decide if the agent has a random chance to slip or not')
parser.add_argument('--map_size', type=str, default='4x4', help='How large you want the map to be in the format NxN')
args = parser.parse_args()



# Create the 4x4 Frozen Lake environment
env = gym.make("FrozenLake-v1", map_name=args.map_size, is_slippery=args.is_slippery)

# Initialize parameters
policy = {}
Rewards = {}
Transitions = {}
visitCounts = {}
learningThreshold = 100 # the minimum number of times a state action pair is visited before it's "known"
discountFactor = 0.99
num_actions = env.action_space.n
num_states = env.observation_space.n
maxReward = 1
epoch = args.iterations / 10
rewards = []


# Initialize Q-table and the state actions pair counter with zeros
Rewards = np.full((num_states + 1, num_actions), maxReward)
Transitions = np.full((num_states + 1, num_actions, num_states), 1 / num_states)
Transitions[-1, 0:num_actions, :] = 0
Transitions[:, :, -1] = 1  # Point all transitions to the fictitious state at the start.
visitCounts = np.zeros((num_states + 1, num_actions), dtype=int)
policy = np.zeros((num_states + 1, num_actions))


# Define a function for value iteration
def value_iteration(Q, T, R, discount_factor, threshold=1e-3):
	while True:
		delta = 0
		for state in range(num_states):
			for action in range(num_actions):
				# Calculate the expected value for the state action pair
				qValue = R[state, action] + discount_factor * sum(T[state, action, sNext] * np.max(Q[sNext]) for sNext in range(num_states))
				delta = max(delta, abs(Q[state, action] - qValue))
				Q[state, action] = qValue
		if delta < threshold:
			break
	return Q


# this is the for loop we're going to be running the training in
for episode in tqdm.tqdm(range(args.iterations)):
	# Reset the environment to start a new episode
	state, info = env.reset()

	done = False
	while not done:

		# chose which action to take
		if np.min(visitCounts[state]) < learningThreshold:
			action = np.argmax(visitCounts[state] < learningThreshold)
		else:
			action = np.argmax(policy[state])

		nextState, reward, done, truncated, info = env.step(action)

		# Update visit counts and Rewards and Transition models
		visitCounts[state, action] += 1
		Rewards[state, action] += (reward - Rewards[state, action]) / visitCounts[state, action]
		Transitions[state, action, :] *= (visitCounts[state, action] - 1) / visitCounts[state, action]
		Transitions[state, action, nextState] += 1 / visitCounts[state, action]
		
		# Update the policy
		policy = value_iteration(policy, Transitions, Rewards, discountFactor)


		# Print information about the current step
		# print(f"state: {state}, Reward: {reward}, Done: {done}, Info: {info}")
	

	# if episode % epoch == 0:
	# 	averageReward = 0
	# 	for _ in range(100):
	# 		state, info = env.reset()

	# 		done = False
	# 		while not done:
	# 			# Implenent the policy
	# 			action = np.argmax(policy[state, :])

	# 			# Step the environment with the chosen action
	# 			state, reward, done, truncated, info = env.step(action)
			
	# 		averageReward += reward

	# 	rewards.append(averageReward / 100)



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
env = gym.make("FrozenLake-v1", map_name=args.map_size, is_slippery=args.is_slippery)

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
	env = gym.make("FrozenLake-v1", render_mode="human", map_name=args.map_size, is_slippery=args.is_slippery)

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
