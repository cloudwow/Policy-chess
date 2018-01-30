import fnmatch
import os
import numpy as np
import chess.pgn
import encoder


def replace_tags(board):
    board_san = board.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    for i in range(len(board.split(" "))):
        if i > 0 and board.split(" ")[i] != '':
            board_san += " " + board.split(" ")[i]
    return board_san


def find_files(directory, pattern='*.pgn'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_text(directory):
    rows = set()
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory)
    for filename in files:
        rows = load_one_file(filename, rows)
        yield filename


total_rows_count = 0


def load_one_file(filename, rows):
    global total_rows_count
    pgn = open(filename)
    for offset, headers in chess.pgn.scan_headers(pgn):
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        result = game.headers["Result"]
        if result == "1-0":
            value = 1.0
        elif result == "0-1":
            value = 0.0
        elif result == "1/2-1/2":
            value = 0.5
        else:
            print "game has no winner: " + result
            continue
        node = game
        while not node.is_end():
            board_fen = replace_tags(node.board().fen())
            next_node = node.variation(0)

            one_hot_index = encoder.one_hot_index_from_move(next_node.move)
            row = board_fen + ":" + str(one_hot_index) + ":" + str(value)
            rows.add(row)
            total_rows_count += 1
            node = next_node
            if len(rows) >= 10000:
                out_name = "examples/examples_{0:08d}.txt".format(
                    total_rows_count)
                print("Saving file: " + out_name)
                np.savetxt(
                    out_name, list(rows), delimiter="", newline="\n", fmt="%s")
                rows = set()

    pgn.close()
    return rows


def main():
    iterator = load_generic_text("./datasets")
    for filename in iterator:
        print("Done with ", filename)
    #load_one_file("./datasets/all.pgn")


if __name__ == '__main__':
    main()
