import sys
import numpy as np
from termcolor import colored
import chess

import encoder


def string_rep(board):
    return board.fen()


def game_value(board, player):
    if board.result() == "1-0":
        result = 1.0
    elif board.result() == "0-1":
        result = -1.0
    else:
        result = board_value(board)

    if player == chess.BLACK:
        result = -result
    return result


def winner(board):
    if board.is_game_over():
        if board.result == "1-0":
            return chess.WHITE
        elif board.result == "0-1":
            return chess.BLACK
        else:
            if board_value(board) > 0:
                return chess.WHITE
            if board_value(board) < 0:
                return chess.BLACK
            else:
                return None


def is_game_over(board):
    return board.is_game_over() or board.fullmove_number > 50


def valid_moves_vector(board):
    result = np.zeros(4096)
    for move in board.legal_moves:
        result[encoder.one_hot_index_from_move(move)] = 1

    return result


def board_value(board):
    value = 0.0
    for x in range(0, 8):
        for y in range(0, 8):
            piece = board.piece_at(chess.square(x, y))
            piece_value = 0.0
            if piece == None:
                piece_value = 0.0
            elif piece.piece_type == chess.PAWN:
                piece_value = 1.0
            elif piece.piece_type == chess.ROOK:
                piece_value = 6.0
            elif piece.piece_type == chess.KNIGHT:
                piece_value = 4.5
            elif piece.piece_type == chess.BISHOP:
                piece_value = 4.5
            elif piece.piece_type == chess.QUEEN:
                piece_value = 9.0
            elif piece.piece_type == chess.KING:
                piece_value = 9999.0

            if piece and piece.color == chess.BLACK:
                piece_value = -piece_value
            value += piece_value
    # normalize within [-1,1]
    return value / 50.0


def dump(board):
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
