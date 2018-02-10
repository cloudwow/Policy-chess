from __future__ import print_function

import os
import fnmatch
import numpy as np
import random
import atexit
import model
import constants
import chess


class RandomBot:

    def get_move(self, board):
        return random.choice(list(board.legal_moves))
