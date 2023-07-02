import math

import numpy as np
import torch as t
from torch.nn import functional as F

from src.game import Board, apply_best_moves, generate_all_games


def gen_games(gametype="all"):
    board = [Board()]

    if gametype == "all":
        print("Generating all possible games...")
        games = generate_all_games(board)

    elif gametype == "strat":
        print("Generating strategic games...")
        games = apply_best_moves(board)

    print(f"Generated {len(games)} games")

    moves = t.tensor(
        [
            [10] + game.moves_played + ([9] * (10 - len(game.moves_played)))
            for game in games
        ],
        requires_grad=False,
    )

    print("Generated array of moves")

    return games, moves


def gen_data_labels(moves):
    data = moves[:, :-1]
    labels = moves[:, 1:]
    print(labels.shape)
    print("Generated data and labels")
    return data, labels


def gen_data_labels_one_hot(labels):
    encoded_labels = F.one_hot(labels).to(t.float)
    print("One hot encoded labels")
    return encoded_labels


def train_test_split(data, labels, split_ratio=0.8, device=None, seed=None):
    if seed is not None:
        t.random.manual_seed(seed)

    assert len(data) == len(labels), "data and label lengths don't match"
    assert isinstance(split_ratio, float)
    assert 0 <= split_ratio <= 1

    inds = t.randperm(len(data))
    split = math.floor(split_ratio * len(data))
    train_inds, test_inds = inds[:split], inds[split:]

    if device is not None:
        data = data.to(device)
        labels = labels.to(device)

    return (data[train_inds], labels[train_inds], data[test_inds], labels[test_inds])


def gen_data(gametype, split_ratio=0.8, device=None, seed=None):
    _, moves = gen_games(gametype)
    data, labels = gen_data_labels(moves)
    encoded_labels = gen_data_labels_one_hot(labels)
    return train_test_split(
        data, encoded_labels, split_ratio=split_ratio, device=device, seed=seed
    )
