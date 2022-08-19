import torch
# import get_init_board, place_piece, get_valid_moves, is_board_full, is_win
from game import get_init_board, place_piece, get_valid_moves, is_board_full, is_win
from view_board import render
import numpy as np
import random
import math

num_simulations = 100

board = np.array(
        [[ 0,-1,-1,-1, 1, 0,-1],
         [ 0, 1,-1, 1, 1, 0, 1],
         [-1, 1,-1, 1, 1, 0,-1],
         [ 1,-1, 1,-1,-1, 0,-1],
         [-1,-1, 1,-1, 1, 1,-1],
         [-1, 1, 1,-1, 1,-1, 1]] 
         )

def ucb_score(parent, child):
    prior_score = child.prior * math.sqrt(parent.visits) / (child.visits + 1)
    if child.visits > 0:
    	value_score = child.value / child.visits
    else:
    	value_score = 0

    return value_score + prior_score

def dummy_model_predict(board):
	value_head = 0.5
	policy_head = [0.5, 0, 0, 0, 0, 0.5, 0]
	return value_head, policy_head

class Node:
	def __init__(self, prior, turn, state):
		self.prior = prior
		self.turn = turn
		self.state = state
		self.children = {}
		self.value = 0
		self.visits = 0

	def expand(self, action_probs):
		# [0.5, 0, 0, 0, 0, 0.5, 0]
		for action, prob in enumerate(action_probs):
			if prob > 0:
				next_state = place_piece(board=self.state, player=self.turn, action=action)
				self.children[action] = Node(prior=prob, turn=self.turn*-1, state=next_state)

	def select_child(self):
		max_score = -99
		for action, child in self.children.items():
			score = ucb_score(self, child)
			if score > max_score:
				selected_action = action
				selected_child = child
				max_score = score

		return selected_action, selected_child


# initialize root
root = Node(prior=0, turn=1, state=board)

# expand the root
value, action_probs = dummy_model_predict(root.state)
root.expand(action_probs=action_probs)

# iterate through simulations
for _ in range(num_simulations):
	node = root
	search_path = [node]
	# select next child until we reach an unexpanded node
	while len(node.children) > 0:
		action, node = node.select_child()
		search_path.append(node)

	value = None
	# calculate value once we reach a leaf node
	if is_board_full(board=node.state):
		value = 0
	if is_win(board=node.state,player=1):
		value = 1
	if is_win(board=node.state,player=-1):
		value = -1

	if value is None:
		# if game is not over, get value from network and expand
		value, action_probs = dummy_model_predict(node.state)
		node.expand(action_probs)

	# back up the value
	for node in search_path:
		node.value += value
		node.visits += 1

