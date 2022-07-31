import numpy as np
from view_board import draw_board, render

# 1. get_init_board
# 2. place_piece
# 3. get_valid_moves
# 4. is_board_full
# 5. is_win

def get_init_board():
    return np.zeros((6,7))

def place_piece(board, player, action):
    board_copy = np.copy(board)
    row_index = sum(board_copy[:,action] == 0)-1
    board_copy[row_index,action] = player
    return board_copy

def get_valid_moves(board):
    # return [0,1,1,1,0,1,1]
    # where 0 is invalid, 1 is valid
    valid_moves = [0] * 7
    for column in range(7):
        if sum(board[:,column] == 0) > 0:
            valid_moves[column] = 1

    return valid_moves

def is_board_full(board):
    return sum(board.flatten() == 0) == 0

def is_win(board,player):
    # return True if player has won, else return False

    # vertical win
    for column in range(7):
        for row in range(3):
            if board[row,column] == board[row+1,column] == board[row+2,column] == board[row+3,column] == player:
                return True

    # horizontal win
    for row in range(6):
        for column in range(4):
            if board[row,column] == board[row,column+1] == board[row,column+2] == board[row,column+3] == player:
                return True

    # diagonal top left to bottom right
    for row in range(3):
        for column in range(4):
            if board[row,column] == board[row+1,column+1] == board[row+2,column+2] == board[row+3,column+3] == player:
                return True

    # diagonal bottom left to top right
    for row in range(5,2,-1):
        for column in range(4):
            if board[row,column] == board[row-1,column+1] == board[row-2,column+2] == board[row-3,column+3] == player:
                return True
                
    return False
