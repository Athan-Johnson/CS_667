from pettingzoo.classic import connect_four_v3
import numpy as np
env = connect_four_v3.env(render_mode="human") # modified this line
env.reset()
print(env)

def sameToken(pos1,pos2):
	if pos1[0] == pos2[0] and pos1[1] == pos2[1]:
		return True
	else:
		return False
		
#Heuristic will be the length of the longest horizontal streak with an open space either before or after items
def heuristic(obs):
	board = obs["observation"]
	longestStreak = 0
	#Search for horizontal lines going right

	for i in range(len(board)):
		currentStreak = 0
		previousToken = np.array([-1,-1])
		for j in range(len(board[i])):
			if not sameToken(board[i][j], previousToken):
				#Open space to the right
				previousToken = board[i][j]
				if sameToken(board[i][j],[0,0]) and currentStreak > longestStreak:
					longestStreak = currentStreak
				currentStreak = 0
			if sameToken(board[i][j],[1,0]):
				currentStreak += 1

	#Search for horizontal lines going left
	for i in range(len(board)):
		currentStreak = 0
		previousToken = np.array([-1,-1])
		for j in reversed(range(len(board[i]))):
			if not sameToken(board[i][j], previousToken):
				#Open space to the right
				previousToken = board[i][j]
				if sameToken(board[i][j],[0,0]) and currentStreak > longestStreak:
					longestStreak = currentStreak
				currentStreak = 0
			if sameToken(board[i][j],[1,0]):
				currentStreak += 1
	return longestStreak
 

def randomAgent(_, agent_):
	return env.action_space(agent_).sample(mask)
 
 
 
MAX_STEPS = 10.000

def recursiveMiniMax(env__, step, amMaximizingPlayer, agent__):
	if step == MAX_STEPS or env__.terminations[agent__] or env__.truncations[agent__]:
		return heuristic(env__.observe(agent__)), step + 1, None

	if amMaximizingPlayer:
		maxEval = -np.inf
		bestAction = None
		for action_ in env__.action_space(agent__):
			envCopy = env__.copy()
			envCopy.step(action_)
			evaluation, step, _ = recursiveMiniMax(envCopy, step, False, envCopy.agent_selection)
			if evaluation > maxEval:
				maxEval = evaluation
				bestAction = action_
		return maxEval, bestAction

	else:
		minEval = np.inf
		bestAction = None
		for action_ in env__.action_space(agent__):
			envCopy = env__.copy()
			envCopy.step(action_)
			evaluation, step, _ = recursiveMiniMax(envCopy, step, True, envCopy.agent_selection)
			if evaluation < minEval:
				minEval = evaluation
				bestAction = action_
		return minEval, bestAction



def miniMax(env_, agent_):
	step = 0
	_, _, ans = recursiveMiniMax(env_, step, True, agent_)
	return ans

	
	
for agent in env.agent_iter():
	print(agent)
	observation, reward, termination, truncation, info = env.last()
	if termination or truncation:
		action = None
		break
	else:
		mask = observation["action_mask"]

		# this is where you would insert your policy
		if agent == "player_0":
			action = miniMax(env, agent)
		else:
			action = randomAgent(env, agent)

		
	env.step(action)
	print(observation)
	print(heuristic(observation))
	


env.close()
