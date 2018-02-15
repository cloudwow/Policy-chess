import math
import numpy as np

import boards
import encoder


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, num_sims, cpuct):
        self.nnet = nnet
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        #print("TREEBOT...")

        for i in range(self.num_sims):
            self.search(canonicalBoard)

        s = boards.string_rep(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(4096)
        ]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = boards.string_rep(canonicalBoard)
        if s not in self.Es:

            if boards.is_game_over(canonicalBoard):
                winner = boards.winner(canonicalBoard)
                if winner == canonicalBoard.turn:
                    self.Es[s] = 1.0
                elif winner:
                    self.Es[s] = -1.0
                else:
                    self.Es[s] = 0.000000001  #small non-zero for draw
            else:
                self.Es[s] = 0

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = boards.valid_moves_vector(canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            self.Ps[s] /= np.sum(self.Ps[s])  # renormalize

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(4096):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s,
                                  a)] + self.cpuct * self.Ps[s][a] * math.sqrt(
                                      self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s])  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        best_move = encoder.move_from_one_hot_index(a)
        # print("best move: " + str(best_move) + " at #" +
        #       str(canonicalBoard.fullmove_number))
        canonicalBoard.push(best_move)

        v = self.search(canonicalBoard)
        canonicalBoard.pop()
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v