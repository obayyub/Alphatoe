import numpy as np
import torch as t
from torch import Tensor
from torch import functional as F

from src.game import Board, apply_best_moves, generate_all_games


def generate_strat_games_data():
    board = [Board()]
    print("Generating stategic games...")
    games = apply_best_moves(board)
    print("Generated " + str(len(games)) + " games")
    moves = np.array([[10] + game.moves_played + [9] for game in games])
    print("Generated array of moves")
    return games, moves


def generate_all_games_data(cfg):
    board = [Board()]
    print("Generating all games...")
    games = generate_all_games(board)
    print("Generated " + str(len(games)) + " games")
    moves = np.array(
        [
            [10] + game.moves_played + ([9] * (cfg.n_ctx - len(game.moves_played)))
            for game in games
        ]
    )
    print("Generated array of moves")
    return games, moves


def gen_data_labels(moves):
    data = np.array(moves[:, :-1])
    labels = moves[:, 1:]
    print(labels.shape())
    print("Generated data and labels")
    return data, labels


def gen_data_labels_one_hot(labels):
    encoded_labels = np.array(F.one_hot(t.tensor(labels)))
    print("One hot encoded labels")
    return encoded_labels


def _train_test_split(data, encoded_labels, cfg, test_train_split=0.8):
    data = t.from_numpy(data)
    encoded_labels = t.from_numpy(encoded_labels).to(t.float)
    total_data = list(zip(data, encoded_labels))
    num_samples = len(total_data)
    train_size = int(test_train_split * num_samples)
    test_size = num_samples - train_size
    split_data = list(t.utils.data.random_split(total_data, [train_size, test_size]))
    train_pairs = split_data[0]
    test_pairs = split_data[1]
    train_data, train_labels = zip(*train_pairs)
    test_data, test_labels = zip(*test_pairs)

    train_data = t.stack(train_data).to(cfg.device)
    train_labels = t.stack(train_labels).to(cfg.device)
    test_data = t.stack(test_data).to(cfg.device)
    test_labels = t.stack(test_labels).to(cfg.device)

    print("Split data and labels into train and test sets")

    return train_data, train_labels, test_data, test_labels


def do_it_all(games_type):
    if games_type == "all":
        _, moves = generate_all_games_data()
    elif games_type == "strat":
        _, moves = generate_strat_games_data()
    data, labels = gen_data_labels(moves)
    encoded_labels = gen_data_labels_one_hot(labels)
    train_data, train_labels, test_data, test_labels = _train_test_split(
        data, encoded_labels
    )
    return train_data, train_labels, test_data, test_labels


def loss_fn(logits: Tensor, labels: Tensor):
    return t.nn.functional.cross_entropy(logits, labels)
