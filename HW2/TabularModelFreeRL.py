import gymnasium as gym

# Create the 4x4 Frozen Lake environment
env = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4", is_slippery=True)

# Reset the environment to start a new episode
obs, info = env.reset()

done = False
while not done:
    # Render the current state of the environment
    env.render()

    # Random action (replace with your action strategy)
    action = env.action_space.sample()

    # Step the environment with the chosen action
    obs, reward, done, truncated, info = env.step(action)

    # Print information about the current step
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

# Close the environment when finished
env.close()

