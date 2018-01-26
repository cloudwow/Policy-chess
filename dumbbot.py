from __future__ import print_function

import os
import fnmatch
import numpy as np
import random
import atexit
import model
import constants
import chess


def board_value(board, color):
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

            if piece and piece.color != color:
                piece_value = -piece_value
            value += piece_value
    value += random.random() * 0.4
    return value


def recurse_value(board, depth=0, max_depth=4):
    #print((" " * depth) + str(depth))

    whose_turn = board.turn
    best_move = None
    best_value = -99999.0
    count = 0
    value = 0
    if depth == max_depth:
        return None, board_value(board, whose_turn)
    for move in board.legal_moves:
        board.push(move)
        _, value = recurse_value(board, depth + 1, max_depth)
        value = -value
        if best_move == None or value > best_value:
            best_move = move
            best_value = value
        board.pop()
        count += 1
    return best_move, value


def get_move(board, max_depth=2):
    move, _ = recurse_value(board, depth=0, max_depth=max_depth)
    return move
