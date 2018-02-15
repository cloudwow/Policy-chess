import unittest

import chess
from tree_bot import Node


class TestStringMethods(unittest.TestCase):

    def test_smoke(self):
        board = chess.Board()
        root_node = Node(board, None, None)
        for i in xrange(25):
            root_node.expand(2.0)

    def test_expand(self):
        board = chess.Board()
        root_node = Node(board, None, None)
        self.assertEqual(root_node.games_evaluated, 0.0)
        root_node.expand(2.0)
        self.assertEqual(root_node.games_evaluated, 1)
        root_node.expand(2.0)
        self.assertEqual(root_node.games_evaluated, 2)


if __name__ == '__main__':
    unittest.main()
