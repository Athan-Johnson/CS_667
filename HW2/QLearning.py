import gymnasium as gym
import random
import tqdm
import argparse


# Parsing
parser = argparse.ArgumentParser(description='Run Q-Learning on the icy lake.')
parser.add_argument('--iterations', default=50000, help='The number of iterations to train for, default is 500,000')
parser.add_argument('--show_final_policy', default=False, help='Decide whether or not to show five games at the end for the user to watch')
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

# Initialize policy table with zeros
for j in range(16):  # Assuming a 4x4 Frozen Lake (16 states)
    for k in range(4):  # 4 possible actions (up, down, left, right)
        policy[j, k] = 0

# this is the for loop we're going to be running the training in
for i in tqdm.tqdm(range(int(args.iterations))):
	# Reset the environment to start a new episode
	# the format here for our dict "policy" is obs, or the number assigned
	# to the state we're in, and the move from that state
	obs, info = env.reset()
	prevObs = obs

	done = False
	while not done:

		if epsilon > random.random():
			action = env.action_space.sample()
		else:
			action = max_with_random_tiebreaker((policy[obs, 0], policy[obs, 1], policy[obs, 2], policy[obs, 3]))

		# Step the environment with the chosen action
		obs, reward, done, truncated, info = env.step(action)
		# Print information about the current step
		# print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

		policy[prevObs, action] += learningRate * (reward + discountFactor * max(policy[obs, 0], policy[obs, 1], policy[obs, 2], policy[obs, 3]) - policy[prevObs, action])

		epsilon = epsilon * 0.995
		prevObs = obs

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
		action = max_with_random_tiebreaker((policy[obs, 0], policy[obs, 1], policy[obs, 2], policy[obs, 3]))

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
			action = max_with_random_tiebreaker((policy[obs, 0], policy[obs, 1], policy[obs, 2], policy[obs, 3]))

			# Step the environment with the chosen action
			obs, reward, done, truncated, info = env.step(action)

			# Print information about the current step
			# print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

	# Close the environment when finished
	env.close()