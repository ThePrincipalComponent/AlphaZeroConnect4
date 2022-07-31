import torch
# import get_init_board, place_piece, get_valid_moves, is_board_full, is_win
from game import *
from view_board import render
import random

num_simulations = 10

board = np.array(
        [[ 0,-1,-1,-1, 1, 0,-1],
         [ 0, 1,-1, 1, 1, 0, 1],
         [-1, 1,-1, 1, 1, 0,-1],
         [ 1,-1, 1,-1,-1, 0,-1],
         [-1,-1, 1,-1, 1, 1,-1],
         [-1, 1, 1,-1, 1,-1, 1]] )

def ucb_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score

def dummy_model_predict(board):
	value_head = 0.5
	policy_head = [0.5,0,0,0,0,0.5,0]
	return value_head, policy_head

class Node:
	def __init__(self, prior, turn, state=None):
		self.prior = prior
		self.turn = turn
		self.state = state
		self.children = {}
		self.value = 0

	def expand(self,action_probs):
		for action, probability in enumerate(action_probs):
			if probability != 0:
				next_state = place_piece(board=self.state, player=self.turn, action=action)
				self.children[action] = Node(prior=probability, turn=self.turn*-1, state=next_state)

	def select_action(self):
		# this is the simplest way to select child
		# will be modified later
		
		selected_action = random.choice([0,5])
		selected_child = self.children[selected_action]

		return selected_action, selected_child

# initialize root
root = Node(prior=0,turn=1,state=board)
value, action_probs = dummy_model_predict(root.state)

# expand the root
root.expand(action_probs=action_probs)
for _ in range(num_simulations):
	node = root
	search_path = [node]
	# select next child until we reach an unexpanded node
	while len(node.children) > 0:
		action, node = node.select_action()
		search_path.append(node)

	# now we are at leaf node
	value=None

	if is_board_full(node.state):
		value = 0
	if is_win(node.state,player=1):
		value = 1
	if is_win(node.state,player=-1):
		value=-1

	if value is None:
		# simulated game is not over
		value, action_probs = dummy_model_predict(node.state * node.turn)
		node.expand(action_probs)

	for node in search_path:
		node.value += value

print(root.children[0].state)
print(root.children[0].value)
print(root.children[5].state)
print(root.children[5].value)
