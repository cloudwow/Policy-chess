import numpy as np
import chess

import constants


def one_hot_index_from_move(move):
    from_x = chess.square_file(move.from_square)
    from_y = chess.square_rank(move.from_square)
    to_x = chess.square_file(move.to_square)
    to_y = chess.square_rank(move.to_square)

    return from_x + from_y * 8 + to_x * 64 + to_y * 8 * 64


def move_from_one_hot_index(one_hot_index):
    from_x = one_hot_index & 7
    from_y = (one_hot_index / 8) & 7
    to_x = (one_hot_index / 64) & 7
    to_y = (one_hot_index / 512) & 7

    return chess.Move(chess.square(from_x, from_y), chess.square(to_x, to_y))


def example_move_to_model_input(position_text):
    parts = position_text.split(":")
    board_state = (parts[0]).replace("/", "")
    move_state = (parts[1]).replace("\n", "")
    value_state = (parts[2]).replace("\n", "")
    move_label = np.zeros(constants.LABEL_SIZE)
    move_label[int(move_state)] = 1.0
    #label = tf.one_hot([int(label_state)], 4096)
    value_label = float(value_state)
    # All pieces plane
    board_pieces = list(board_state.split(" ")[0])
    board_pieces = [ord(val) for val in board_pieces]
    board_pieces = np.reshape(board_pieces,
                              (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
    # Only spaces plane
    board_blank = [int(val == '1') for val in board_state.split(" ")[0]]
    board_blank = np.reshape(board_blank,
                             (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
    # Only white plane
    board_white = [int(val.isupper()) for val in board_state.split(" ")[0]]
    board_white = np.reshape(board_white,
                             (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
    # Only black plane
    board_black = [
        int(not val.isupper() and val != '1')
        for val in board_state.split(" ")[0]
    ]
    board_black = np.reshape(board_black,
                             (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
    # One-hot integer plane current player turn
    current_player = board_state.split(" ")[1]
    current_player = np.full(
        (constants.IMAGE_SIZE, constants.IMAGE_SIZE),
        int(current_player == 'w'),
        dtype=int)
    # One-hot integer plane extra data
    extra = board_state.split(" ")[4]
    extra = np.full(
        (constants.IMAGE_SIZE, constants.IMAGE_SIZE), int(extra), dtype=int)
    # One-hot integer plane move number
    move_number = board_state.split(" ")[5]
    move_number = np.full(
        (constants.IMAGE_SIZE, constants.IMAGE_SIZE),
        int(move_number),
        dtype=int)
    # Zeros plane
    zeros = np.full((constants.IMAGE_SIZE, constants.IMAGE_SIZE), 0, dtype=int)

    planes = np.vstack((np.copy(board_pieces), np.copy(board_white),
                        np.copy(board_black), np.copy(board_blank),
                        np.copy(current_player), np.copy(extra),
                        np.copy(move_number), np.copy(zeros)))
    planes = np.reshape(
        planes,
        (constants.IMAGE_SIZE, constants.IMAGE_SIZE, constants.FEATURE_PLANES))
    return (planes, move_label, value_label)
