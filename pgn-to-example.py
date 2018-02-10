import fnmatch
import os
import numpy as np
import chess.pgn


def find_files(directory, pattern='*.pgn'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_text(directory):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory)
    labels = set()
    for filename in files:
        k = 0
        pgn = open(filename)
        for offset, headers in chess.pgn.scan_headers(pgn):
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                label_san = node.board().san(next_node.move)

                if label_san not in labels:
                    labels.add(label_san)
                node = next_node
            if k % 100 == 0 and k > 1:
                print("Labeling file: " + filename + ", step: " + str(k) +
                      " label count:" + str(len(labels)))
            if k % 2000 == 0 and k > 1:

                np.savetxt(
                    "labels/labels-partway.txt",
                    np.array(list(labels)),
                    delimiter="",
                    newline="\n",
                    fmt="%s")

            k += 1
        pgn.close()
    np.savetxt(
        "labels/labels.txt",
        np.array(list(labels)),
        delimiter="",
        newline="\n",
        fmt="%s")


def main():
    load_generic_text("./datasets")
    print("Done.")


if __name__ == '__main__':
    main()
