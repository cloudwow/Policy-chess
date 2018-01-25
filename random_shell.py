"""Playing script for the network."""

from __future__ import print_function
import random
import sys
import os
import fnmatch
import numpy as np
import tensorflow as tf
from termcolor import colored
import chess.pgn
import atexit
import constants
import bot


def render_board(board):
    last_move = board.peek()
    for x in range(0, 8):
        print("\n" + ("--" * 16))

        sys.stdout.write("|")
        for y in range(0, 8):
            if x == chess.square_file(
                    last_move.to_square) and y == chess.square_rank(
                        last_move.to_square):
                fill_char = "*"
            else:
                fill_char = " "
            piece = board.piece_at(chess.square(y, x))
            if piece:
                if piece.color == chess.WHITE:
                    sys.stdout.write(
                        fill_char + colored(str(piece), "white") + fill_char)
                else:
                    sys.stdout.write(
                        fill_char + colored(str(piece), "green") + fill_char)
            else:
                sys.stdout.write(fill_char * 3)
            sys.stdout.write("|")

    print("\n" + ("-+" * 16))


def main():
    print('\nPlaying...\nComputer plays white.\n')
    board = chess.Board()
    quit = False
    while (not quit and not board.is_game_over()):
        # We get the movement prediction
        move = bot.get_move(board)

        print('The computer wants to move to:', move)
        board.push(move)
        legal_moves = list(board.legal_moves)
        index = 0
        render_board(board)
        # we move now
        moved = False
        move = random.choice(legal_moves)
        board.push(move)
        print(board)

    print("\nEnd of the game.")
    print("Game result:")
    print(board.result())


def quit_gracefully():
    print('Bye')


atexit.register(quit_gracefully)

if __name__ == '__main__':
    main()
