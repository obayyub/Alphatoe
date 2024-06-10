from alphatoe import models, plot, interpretability, game
import pandas as pd
import torch
import argparse
import einops
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import numpy as np
import tqdm
import random
import pickle

move_count = 8
# load alphatoe model and trained 512 SAE

autoenc = models.SparseAutoEncoder(512, 512).cuda()
autoenc.load_state_dict(
    torch.load(
        "scripts/models/SAE_hidden_size-512_lamda-2.5e-08_batch_4096_wo_l1_cosine-0.pt"
    )
)
model = interpretability.load_model(
    "./scripts/models/prob all 8 layer control-20230718-185339"
)


def neuron_activations(seq):
    def hook(module, input, output):
        result = output.clone()
        module.captured_activations = result

    try:
        with torch.no_grad():
            handle = model.blocks[0].mlp.hook_post.register_forward_hook(hook)
            _ = model(seq)
            activations = model.blocks[0].mlp.hook_post.captured_activations
            handle.remove()
    except Exception as e:
        handle.remove()
        raise e

    return activations


def get_random_sequences(seqs, condition, num_samples=2000):
    filtered_seqs = [seq for seq in seqs if condition(seq)]
    print(f"filtered {len(filtered_seqs)} sequences")
    return random.sample(filtered_seqs, num_samples)


boards = game.generate_all_games([game.Board()])

move_sequences = torch.stack(
    [
        torch.tensor([10] + board.moves_played)
        for board in boards
        if len(board.moves_played) == move_count
    ]
)

# wow this is definitely code smell
moves_by_content = {
    content: {
        "contains": torch.stack(
            get_random_sequences(
                move_sequences, lambda x, c=content: c in x and c != x[-1]
            )
        ),
        "not_contains": torch.stack(
            get_random_sequences(
                move_sequences, lambda x, c=content: c not in x and c != x[-1]
            )
        ),
    }
    for content in range(9)
}

print("generated boards move number dict")
# change this so it takes last move and second to last move
features_by_content = {
    content: {
        key: {
            "idx -1 activations": autoenc.get_activations(
                neuron_activations(tensor_data)
            )[:, -1],
            "idx -2 activations": autoenc.get_activations(
                neuron_activations(tensor_data)
            )[:, -2],
        }
        for key, tensor_data in content_data.items()
    }
    for content, content_data in moves_by_content.items()
}

print("generated activations by content dict")

# save the json
with open(f"SAE_features_by_token_{move_count}_move_games_512_lamda-2.5e-08_epoch-600_batch_4096_coslam_0.pkl", "wb") as f:
    pickle.dump(features_by_content, f)

with open(f"{move_count}_move_games.pkl", "wb") as f:
    pickle.dump(moves_by_content, f)
