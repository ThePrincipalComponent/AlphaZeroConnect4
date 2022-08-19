import pygame
import numpy as np
import sys

def draw_board(screen,board):
    COLUMN_COUNT = 7
    ROW_COUNT = 6
    SQUARESIZE = 100
    RADIUS = int(SQUARESIZE/2 - 5)

    BLUE = (52, 186, 235)
    GREY = (70, 71, 70)
    WHITE = (255,255,255)
    YELLOW = (230,230,20)

    width = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT+1) * SQUARESIZE

    size = (width, height)
    board = np.flip(board,0)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, GREY, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, WHITE, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, BLUE, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == -1:
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

def render(board):
    pygame.init()
    screen = pygame.display.set_mode((700,700))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            draw_board(screen,board)
            pygame.display.update()