"""Playing script for the network."""

from __future__ import print_function
import random
import sys
import os
import time
import fnmatch
import numpy as np
import tensorflow as tf
from termcolor import colored
import chess.pgn
import atexit
import constants
import bot
import dumbbot


def render_board(board):
    last_move = board.peek()
    for y in range(7, -1, -1):
        print("\n" + ("--" * 16))

        sys.stdout.write("|")
        for x in range(0, 8):
            fill_char = " "

            if x == chess.square_file(
                    last_move.to_square) and y == chess.square_rank(
                        last_move.to_square):
                fill_char = "*"
            if x == chess.square_file(
                    last_move.from_square) and y == chess.square_rank(
                        last_move.from_square):
                fill_char = "#"
            piece = board.piece_at(chess.square(x, y))

            if piece:
                if piece.color == chess.WHITE:
                    draw_color = "white"
                else:
                    draw_color = "green"

                sys.stdout.write(
                    colored(fill_char + str(piece) + fill_char, draw_color))

            else:
                if board.turn == chess.BLACK:
                    draw_color = "white"
                else:
                    draw_color = "green"

                sys.stdout.write(colored(fill_char * 3, draw_color))
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
        index = 0
        render_board(board)
        time.sleep(2)

        if not board.is_game_over():
            # we move now
            moved = False
            move = bot.get_move(board)
            board.push(move)
            render_board(board)

        # print(board)

    print("\nEnd of the game.")
    print("Game result:")
    print(board.result())


def quit_gracefully():
    print('Bye')


atexit.register(quit_gracefully)

if __name__ == '__main__':
    main()
