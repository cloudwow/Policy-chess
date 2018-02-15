from __future__ import division

import sys
import time
from math import log, sqrt
import random

import chess

import boards


class Node(object):

    def __init__(self, board, parent_node, move=None):
        if move:
            board.push(move)
        self.fen = board.fen()
        self.move = move
        self.my_player = board.turn
        self.parent_node = parent_node

        if boards.is_game_over(board):
            self.game_over_value = boards.game_value(board, self.my_player)
        else:
            self.game_over_value = None
        self.games_evaluated = 0.0
        self.total_value = 0.0
        self.children = None
        if move:
            board.pop()

    def expand(self, temperature):
        if self.game_over_value != None:
            self.add_game_value(self.game_over_value)
            return

        if self.games_evaluated == 0:
            # the first time just play out
            self.playout()

        else:
            # after the first we start expanding children
            if self.children == None:
                board = chess.Board(self.fen)
                self.children = [
                    Node(board, self, move=m) for m in board.legal_moves
                ]

            self.choose_best_child(temperature).expand(temperature)

    def choose_best_child(self, temperature):
        # return highest UCB1
        return max(self.children, key=lambda x: x.UCB1(temperature))

    def playout(self):
        board = chess.Board(self.fen)
        i = 0
        while i < 80 and not boards.is_game_over(board):
            board.push(random.choice(list(board.legal_moves)))
            i += 1
        self.add_game_value(-boards.game_value(board, self.my_player))

    def add_game_value(self, game_value):
        self.total_value += game_value
        self.games_evaluated += 1.0
        if self.parent_node:
            self.parent_node.add_game_value(-game_value)

    def average_value(self):
        """ returns average value of explored games """
        if self.games_evaluated == 0:
            return 0.0
        return self.total_value / self.games_evaluated

    def UCB1(self, temperature):
        """ returns average value of explored games """
        if self.games_evaluated == 0:
            return sys.float_info.max
        return self.average_value() + temperature * (
            log(self.parent_node.games_evaluated) / self.games_evaluated)


class TreeBot(object):

    def __init__(self, calculation_time=30, max_actions=4000, temperature=1.4):

        self.calculation_time = calculation_time
        self.max_actions = max_actions

    def get_move(self, board):
        root_node = Node(board, None, None)

        for i in xrange(self.max_actions):
            root_node.expand(2.0)
        return root_node.choose_best_child(0.0).move
