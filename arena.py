import tensorflow as tf
import chess
import boards


class Arena:

    def __init__(self, sess, player_1, player_2):
        self.sess = sess
        self.player_1 = player_1
        self.player_2 = player_2
        self.quit = False

    def play(self, how_many=1):
        player_1_wins = 0
        player_2_wins = 0
        draws = 0

        white_player = self.player_1
        black_player = self.player_2
        for game_index in range(0, how_many):
            if self.sess:
                saver = tf.train.Saver()
            board = chess.Board()

            self.quit = False
            move_index = 0

            while (not self.quit and move_index < 120 and
                   not board.is_game_over()):
                move_index += 1
                # We get the movement prediction
                move = white_player.get_move(board)

                board.push(move)
                boards.dump(board)

                if not board.is_game_over():
                    # we move now
                    moved = False
                    move = black_player.get_move(board)
                    board.push(move)

                    boards.dump(board)

            boards.dump(board)

            print("\nEnd of the game.")
            print("Game result:")
            print(board.result())
            winner = None
            if board.result() == "1-0":
                winner = white_player
                print("Winner is WHITE")
            elif board.result() == "0-1":
                winner = black_player
                print("Winner is BLACK")
            elif board.result() == "*":
                if boards.board_value(board) > 0:
                    winner = white_player

                    print("Winner is white on pieces")
                if boards.board_value(board) < 0:
                    winner = black_player

                    print("Winner is black on pieces")

            else:
                print("GAME IS STALEMATE")
            if winner == self.player_1:
                player_1_wins += 1
            elif winner == self.player_2:
                player_2_wins += 1
            else:
                draws += 1
            print("player_1:{0}, player_2:{1},  draws:{2}".format(
                player_1_wins, player_2_wins, draws))
            derp = white_player
            white_player = black_player
            black_player = derp
