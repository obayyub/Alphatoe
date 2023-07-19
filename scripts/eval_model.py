import argparse
import json
import os

import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer

from alphatoe import evals


def main(args: argparse.Namespace):
    if args.model_name == "recent":
        dir = os.path.dirname(os.path.realpath(__file__))
        model_name = get_most_recent_file(f"{dir}/")
        args = load_config(model_name)
        cfg = create_hooked_transformer_config(args)
        model = HookedTransformer(cfg)
        model = model.load_state_dict(load_weights(model_name))

    else:
        cfg = create_hooked_transformer_config(args)
        model = HookedTransformer(cfg)
    args = load_config(args.model_name)
    weights = load_weights(args.model_name)
    model = HookedTransformer(cfg)
    model.load_state_dict(weights)
    games = evals.sample_games(model, args.temp, args.num_games)
    error_rate = evals.error_rate(games)
    print(f"Error rate is: {error_rate}")


def load_weights(model_name: str):
    return torch.load(f"{model_name}.pt")


def load_config(model_name: str):
    with open(f"{model_name}.json", "r") as f:
        return json.load(f)


def get_most_recent_file(directory: str):
    files = os.listdir(directory)

    timestamp_regex = re.compile(r"(\d{8}-\d{6})\.pt")

    files = [f for f in files if timestamp_regex.search(f)]

    if files == []:
        raise ValueError("You need to have run models before you can fine tune them!")

    files.sort(
        key=lambda f: datetime.strptime(
            timestamp_regex.search(f).group(1), "%Y%m%d-%H%M%S"
        ),
        reverse=True,
    )

    most_recent_file = files[0][:-3]  # The most recent file without extension
    return most_recent_file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_name", type=str)
    ap.add_argument("--num_games", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=1)

    main(ap.parse_args())
