#from pettingzoo.classic import connect_four_v3
import connect_four_v3
import numpy as np
import copy
import random
Env = connect_four_v3.env(render_mode="human")
Env.reset()

print(Env)

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
    


def randomAgent(env_, mask_, agent_, __):
    return env_.action_space(agent_).sample(mask_)


def is_terminal_state(env_):
    # Check if the game has ended due to a win or a draw
    return env_.terminations[env_.agent_selection] or env_.truncations[env_.agent_selection]

def evaluate_board(env_):
    # Heuristic evaluation of the board, you can improve this
    if env_.rewards[env_.agent_selection] == 1:
        return 100  # Winning state
    elif env_.rewards[env_.agent_selection] == -1:
        return -100  # Losing state
    else:
        return heuristic(env_.observe(env_.agent_selection))  # Neutral state

def replay_env(env_, move_history):
    """Reset and replay the environment up to the current point based on move history."""
    env_.reset()
    for move in move_history:
        env_.step(move)


def recursiveMiniMax(env_, depth, is_maximizing, move_history, alpha=-np.inf, beta=np.inf):
    if depth == 0 or is_terminal_state(env_):
        return evaluate_board(env_), None

    observation_, reward_, termination_, truncation_, info_ = env_.last()
    valid_moves = observation_["action_mask"]

    if is_maximizing:
        max_eval = -np.inf
        best_move = None
        for move in range(len(valid_moves)):
            if valid_moves[move]:
                # Create a new move history including this move
                new_move_history = move_history + [move]

                # Reset the environment and replay up to the current state
                replay_env(env_, new_move_history)

                eval_score, _ = recursiveMiniMax(env_, depth - 1, False, new_move_history, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                # Update alpha and prune if necessary
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break # beta cutoff, prune this branch

        return max_eval, best_move

    else:  # Minimizing player
        min_eval = np.inf
        best_move = None
        for move in range(len(valid_moves)):
            if valid_moves[move]:
                # Create a new move history including this move
                new_move_history = move_history + [move]

                # Reset the environment and replay up to the current state
                replay_env(env_, new_move_history)

                eval_score, _ = recursiveMiniMax(env_, depth - 1, True, new_move_history, alpha, beta)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                # Update beta and prune if necessary
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break # alpha cutoff, prune this branch

        return min_eval, best_move



def miniMax(_, __, ___, actions_):
    depth = 4
    envCopy = connect_four_v3.env()
    envCopy.reset()
    for a in actions_:
        envCopy.step(a)
    _, ans = recursiveMiniMax(envCopy, depth, True, actions_)
    return ans


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = [None] * 7
        self.wins = 0
        self.visits = 0
        observation_, reward_, termination_, truncation_, info_ = state.last()
        self.untried_actions = np.nonzero(observation_["action_mask"])[0]
        self.untried_actions = self.untried_actions.tolist()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.4):
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
            if child is not None else -1
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def best_move(self):
        # Return the child with the highest win rate (no exploration factor)
        choices_weights = [(child.wins / child.visits) if child is not None else -1 for child in self.children]
        return np.argmax(choices_weights)

    def expand(self, action_history):
        action_ = self.untried_actions.pop()
        next_state = connect_four_v3.env()
        next_state.reset()
        for a in action_history:
            next_state.step(a)
        next_state.step(action_)
        child_node = Node(next_state, parent=self)
        self.children[action_] = child_node
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_terminal_node(self):
        return is_terminal_state(self.state)


def uct_search(root_state, actions):
    max_iter = 10000
    root_node = Node(root_state)

    for _ in range(max_iter):
        node = root_node
        state = connect_four_v3.env()
        state.reset()
        for a in actions:
            state.step(a)

        # Selection
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child()

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand(actions)

        # Simulation
        final_reward = 0
        while not is_terminal_state(state):
            observation_, reward_, termination_, truncation_, info_ = state.last()

            # Get valid actions from the action mask
            valid_actions = np.nonzero(observation_["action_mask"])[0]

            # Randomly select a valid action (you could also use a more strategic choice here)
            action_ = random.choice(valid_actions)

            state.step(action_) # Accumulate reward (you may want to handle multi-agent rewards differently)

            observation_, reward_, termination_, truncation_, info_ = state.last()

            # Check if the game is over
            if termination_ or truncation_:
                final_reward = reward_
                break

        # Backpropagation
        while node is not None:
            node.update(final_reward)  # Update node with the cumulative reward
            node = node.parent

    return root_node.best_move()


def monteCarlo(_, __, ___, actions):
    envCopy = connect_four_v3.env()
    envCopy.reset()
    for a in actions:
        envCopy.step(a)
    return uct_search(envCopy, actions)


# turnCount = 0
# fifthState = ""
# fifthPlayer = ""
actions_history = []
for agent in Env.agent_iter():
    # if turnCount == 5:
    #     fifthState = copy.deepcopy(Env.board)
    #     fifthPlayer = Env.agent_selection
    observation, reward, termination, truncation, info = Env.last()
    if termination or truncation:
        action = None
        break
    else:
        mask = observation["action_mask"]

        # this is where you would insert your policy
        if agent == "player_0":
            action = miniMax(Env, mask, agent, actions_history)
        else:
            action = monteCarlo(Env, mask, agent, actions_history)

    Env.step(action)
    actions_history = actions_history + [action]
    print(observation)

    player_1_reward = Env.rewards['player_0']  # Player 1 (agent 'player_0')
    player_2_reward = Env.rewards['player_1']  # Player 2 (agent 'player_1')

    if player_1_reward > player_2_reward:
        print("Player 0 wins!")
    elif player_2_reward > player_1_reward:
        print("Player 1 wins!")
    else:
        print("It's a draw!")
    # turnCount += 1
# print("RESET")
# observation = Env.setState(fifthState,fifthPlayer)
# print(observation)
# termination = False
# truncation = False
# print("CONTINUATION")
# while termination is False and truncation is False:
#     agent = Env.agent_selection
#     #print(Env.agent_selection)
#     observation, reward, termination, truncation, info = Env.last()
#     print(termination,truncation)
#     if termination or truncation:
#         action = None
#         break
#     else:
#         mask = observation["action_mask"]
#
#         # this is where you would insert your policy
#         if(agent == "player_0"):
#             action = Env.action_space(agent).sample(mask)
#         else:
#             action = Env.action_space(agent).sample(mask)
#
#     Env.step(action)
#     print(observation)
#     print(heuristic(observation))
#     turnCount += 1
# Env.close()