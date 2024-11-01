import gymnasium as gym
import random
import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np


# Parsing
parser = argparse.ArgumentParser(description='Run Q-Learning on the icy lake.')
parser.add_argument('--iterations', type=int, default=100000, help='The number of iterations to train for, default is 100,000')
parser.add_argument('--show_final_policy', type=bool, default=False, help='Decide whether or not to show five games at the end for the user to watch, default is False')
args = parser.parse_args()





def max_with_random_tiebreaker(lst):
    if not lst:
        raise ValueError("List cannot be empty")

    # Find the maximum value in the list
    max_value = max(lst)

    # Get all indices where the max value occurs
    max_indices = [i for i, x in enumerate(lst) if x == max_value]

    # Randomly select one of the indices with the maximum value
    random_index = random.choice(max_indices)

    # Return the index of the max value
    return random_index



# Create the 4x4 Frozen Lake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)


# Set up Q-learning policy table
policy = {}

# Initialize learning parameters
learningRate = 0.1
discountFactor = 0.9
epsilon = 1.0
epsilonDecay = 1 - (50 / args.iterations)
min_epsilon = 0.1
rewards = []
epoch = args.iterations / 10
num_actions = env.action_space.n
num_states = env.observation_space.n


# Initialize Q-table with zeros
policy = np.zeros((num_states, num_actions))

# this is the for loop we're going to be running the training in
for episode in tqdm.tqdm(range(args.iterations)):
	# Reset the environment to start a new episode
	# the format here for our dict "policy" is state, or the number assigned
	# to the state we're in, and the move from that state
	state, info = env.reset()
	prevState = state

	done = False
	while not done:

		if epsilon > random.random():
			action = env.action_space.sample()
		else:
			action = max_with_random_tiebreaker((policy[state, 0], policy[state, 1], policy[state, 2], policy[state, 3]))

		# Step the environment with the chosen action
		state, reward, done, truncated, info = env.step(action)
		# Print information about the current step
		# print(f"state: {state}, Reward: {reward}, Done: {done}, Info: {info}")

		policy[prevState, action] += learningRate * (reward + discountFactor * max(policy[state, 0], policy[state, 1], policy[state, 2], policy[state, 3]) - policy[prevState, action])

		prevState = state
	
	# Decay epsilon to reduce exploration over time
	epsilon = max(min_epsilon, epsilon * epsilonDecay)

	if episode % epoch == 0:
		averageReward = 0
		for _ in range(100):
			state, info = env.reset()

			done = False
			while not done:
				# Implenent the policy
				action = max_with_random_tiebreaker((policy[state, 0], policy[state, 1], policy[state, 2], policy[state, 3]))

				# Step the environment with the chosen action
				state, reward, done, truncated, info = env.step(action)
			
			averageReward += reward

		rewards.append(averageReward / 100)


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
plt.savefig('Rewards_over_epochs_QL.png')
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
		action = max_with_random_tiebreaker((policy[state, 0], policy[state, 1], policy[state, 2], policy[state, 3]))

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
			action = max_with_random_tiebreaker((policy[state, 0], policy[state, 1], policy[state, 2], policy[state, 3]))

			# Step the environment with the chosen action
			state, reward, done, truncated, info = env.step(action)

			# Print information about the current step
			# print(f"state: {state}, Reward: {reward}, Done: {done}, Info: {info}")

	# Close the environment when finished
	env.close()
