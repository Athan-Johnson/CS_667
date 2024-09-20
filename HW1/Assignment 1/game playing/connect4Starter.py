from pettingzoo.classic import connect_four_v3
import numpy as np
env = connect_four_v3.env()
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
	
	
for agent in env.agent_iter():
	print(agent)
	observation, reward, termination, truncation, info = env.last()
	if termination or truncation:
		action = None
		break
	else:
		mask = observation["action_mask"]

		# this is where you would insert your policy
		if(agent == "player_0"):
			action = env.action_space(agent).sample(mask)
		else:
			action = env.action_space(agent).sample(mask)
		
	env.step(action)
	print(observation)
	print(heuristic(observation))
	


env.close()