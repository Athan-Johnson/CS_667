#from pettingzoo.classic import connect_four_v3
import connect_four_v3
import numpy as np
import copy
env = connect_four_v3.env(render_mode="human")
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


MAX_DEPTH = 10.000

def recursiveMiniMax(env__, depth, actions, amMaximizingPlayer, agent__):
    if depth == 0 or env__.check_for_winner():
        return heuristic(env__.observe(agent__)), -1

    if amMaximizingPlayer:
        maxEval = -np.inf
        bestAction = -1

        envCopy = connect_four_v3.env()
        envCopy.reset()

        for action_ in range(6):
            for a in actions:
                envCopy.step(a)
            envCopy.step(action_)
            actions.append(action_)
            evaluation, _ = recursiveMiniMax(envCopy, depth - 1, actions, False, envCopy.agent_selection)
            if evaluation > maxEval:
                maxEval = evaluation
                bestAction = action_
            actions.pop()
            envCopy.reset()
        return maxEval, bestAction

    else:
        minEval = np.inf
        bestAction = -1

        envCopy = connect_four_v3.env()
        envCopy.reset()

        for action_ in range(6):
            for a in actions:
                envCopy.step(a)
            envCopy.step(action_)
            actions.append(action_)
            evaluation, _ = recursiveMiniMax(envCopy, depth - 1, actions, True, envCopy.agent_selection)
            if evaluation < minEval:
                minEval = evaluation
                bestAction = action_
            actions.pop()
            envCopy.reset()
        return minEval, bestAction



def miniMax(env_, agent_):
    depth = 2
    actions = []
    _, ans = recursiveMiniMax(env_, depth, actions, True, agent_)
    return ans

turnCount = 0
fifthState = ""
fifthPlayer = ""
for agent in env.agent_iter():
    if turnCount == 5:
        fifthState = copy.deepcopy(env.board)
        fifthPlayer = env.agent_selection
    print(agent)
    print(env.agent_selection)
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
        break
    else:
        mask = observation["action_mask"]

        # this is where you would insert your policy
        if(agent == "player_0"):
            action = randomAgent(env, agent)
        else:
            action = miniMax(env, agent)

    env.step(action)
    print(observation)
    print(heuristic(observation))
    turnCount += 1
# print("RESET")
# observation = env.setState(fifthState,fifthPlayer)
# print(observation)
# termination = False
# truncation = False
# print("CONTINUATION")
# while termination is False and truncation is False:
#     agent = env.agent_selection
#     #print(env.agent_selection)
#     observation, reward, termination, truncation, info = env.last()
#     print(termination,truncation)
#     if termination or truncation:
#         action = None
#         break
#     else:
#         mask = observation["action_mask"]
#
#         # this is where you would insert your policy
#         if(agent == "player_0"):
#             action = env.action_space(agent).sample(mask)
#         else:
#             action = env.action_space(agent).sample(mask)
#
#     env.step(action)
#     print(observation)
#     print(heuristic(observation))
#     turnCount += 1
# env.close()