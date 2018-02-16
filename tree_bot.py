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
        self.children = None

        if boards.is_game_over(board):
            self.game_over_value = -boards.game_value(board, self.my_player)
        else:
            self.game_over_value = None
            self.games_evaluated = 1.0
            self.total_value = -boards.board_value(board, self.my_player)
        if move:
            board.pop()

    def expand(self, temperature):
        if self.game_over_value != None:
            self.add_game_value(self.game_over_value)
            return

        if self.games_evaluated <= 1:
            # the first time just play out
            self.playout()

        else:
            # after the first we start expanding children
            if self.children == None:
                board = chess.Board(self.fen)
                self.children = [
                    Node(board, self, move=m) for m in board.legal_moves
                ]

            self.choose_child_for_expand(temperature).expand(temperature)

    def choose_child_for_expand(self, temperature):
        # return highest UCB1
        return max(
            self.non_terminal_children(), key=lambda x: x.UCB1(temperature))

    def choose_best_child(self):
        # return highest UCB1
        return max(self.children, key=lambda x: x.UCB1(0.0))

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
        if self.is_terminal_node():
            return self.game_over_value
        if self.games_evaluated == 0:
            return sys.float_info.max
        return self.average_value() + temperature * (
            log(self.parent_node.games_evaluated) / self.games_evaluated)

    def non_terminal_children(self):
        return filter(lambda x: x.is_terminal_node() == False, self.children)

    def is_terminal_node(self):
        return self.game_over_value != None


class TreeBot(object):

    def __init__(self, calculation_time=10, max_actions=4000, temperature=1.4):

        self.calculation_time = calculation_time
        self.max_actions = max_actions

    def get_move(self, board):
        root_node = Node(board, None, None)
        now = start_time = time.time()
        while now - start_time < 15.0:
            root_node.expand(0.3)
            now = time.time()
        return root_node.choose_best_child().move
