"""Training script for the network."""

from __future__ import print_function

import os
import random
import numpy as np

import constants
import encoder


def generate_batches(directory, batch_size=constants.BATCH_SIZE):
    '''Generator that yields text raw from the directory.'''
    while True:
        files = _find_files(directory)
        np.random.shuffle(files)
        for filename in files:
            text = _read_text(filename, batch_size)
            yield reformat(text)


def _find_files(directory):
    result = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            result.append(os.path.join(directory, file))
    return result


def _read_text(filename, batch_size):
    with open(filename) as f_in:
        lines = (line.rstrip() for line in f_in)
        # drop empty lines
        lines = list(line for line in lines if line)
        return np.random.choice(lines, batch_size)


def reformat(games):
    for game in games:
        yield encoder.example_move_to_model_input(game)
