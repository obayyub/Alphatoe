import numpy as np
import torch
from torch import Tensor
from transformer_lens import HookedTransformer, HookedTransformerConfig
import json
import matplotlib.pyplot as plt
from copy import copy
from typing import Optional


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def numpy(t):
    return t.cpu().detach().numpy()


def load_model(model_file_name: str):
    weights = torch.load(model_file_name + ".pt", map_location=get_device())
    with open(model_file_name + ".json", "r") as f:
        args = json.load(f)
    model_cfg = HookedTransformerConfig(
        n_layers=args["n_layers"],
        n_heads=args["n_heads"],
        d_model=args["d_model"],
        d_head=args["d_head"],
        d_mlp=args["d_mlp"],
        act_fn=args["act_fn"],
        normalization_type=args["normalization_type"],
        d_vocab=11,
        d_vocab_out=10,
        n_ctx=10,
        init_weights=True,
        device=get_device(),
        seed=args["seed"],
    )
    model = HookedTransformer(model_cfg)
    model.load_state_dict(weights)
    return model


def ablate_all_but_one_head(model, head, seq):
    def hook(module, input, output):
        result = torch.zeros_like(output)
        result[:, :, head, :] = output[:, :, head, :]
        return result

    model.cfg.use_attn_result = True
    try:
        handle = model.blocks[0].attn.hook_result.register_forward_hook(hook)
        logits = model(torch.tensor(seq))
        handle.remove()
    except Exception as e:
        handle.remove()
        raise e

    return logits


def ablate_one_head(model, head, seq):
    def hook(module, input, output):
        result = output.clone()
        result[:, :, head, :] = 0
        return result

    model.cfg.use_attn_result = True
    try:
        handle = model.blocks[0].attn.hook_result.register_forward_hook(hook)
        logits = model(torch.tensor(seq))
        handle.remove()
    except Exception as e:
        handle.remove()
        raise e

    return logits


def ablate_mlp(model, seq):
    def hook(module, input, output):
        return torch.zeros_like(output)

    try:
        handle = model.blocks[0].mlp.register_forward_hook(hook)
        logits = model(torch.tensor(seq))
        handle.remove()
    except Exception as e:
        handle.remove()
        raise e

    return logits


def plot_predictions(seq, logits, suptitle="", show_vals=False, **kwargs):
    plt.figure(figsize=(14, 5))
    plt.gcf().set_facecolor("white")
    show_vals = show_vals
    preds = torch.softmax(logits, axis=-1)
    plt.subplot(1, 2, 1)
    plt.imshow(numpy(logits)[0], cmap="jet", **kwargs)
    plt.yticks(np.arange(logits.shape[1]), labels=seq)
    plt.ylabel("Current token", fontsize=15)
    plt.xlabel("Predicted token", fontsize=15)
    plt.title("Logits", fontsize=20)
    plt.colorbar()
    if show_vals:
        plt.rcParams["figure.dpi"] = 100
        for i in range(logits.shape[1]):
            for j in range(logits.shape[2]):
                value = logits[0, i, j].item()
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=8,
                )

    plt.subplot(1, 2, 2)
    plt.imshow(numpy(preds)[0], cmap="jet", vmin=0, vmax=1)
    plt.yticks(np.arange(logits.shape[1]), labels=seq)
    plt.ylabel("Current token", fontsize=15)
    plt.xlabel("Predicted token", fontsize=15)
    plt.title("Preds", fontsize=20)
    plt.colorbar()
    plt.suptitle(suptitle, fontsize=20)
    if show_vals:
        plt.rcParams["figure.dpi"] = 100
        for i in range(preds.shape[1]):
            for j in range(preds.shape[2]):
                value = preds[0, i, j].item()
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=8,
                )
    plt.gcf().set_facecolor("white")
    plt.show()


def get_neuron_acts(model, seq):
    with torch.no_grad():
        logits, cache = model.run_with_cache(torch.tensor(seq))
    neuron_activations = copy(cache["post", 0][0])
    return neuron_activations


def ablate_one_neuron(neuron, seq):
    def hook(module, input, output):
        result = output.clone()
        result[:, :, neuron] = 0
        return result

    try:
        handle = model.blocks[0].mlp.hook_post.register_forward_hook(hook)
        logits = model(torch.tensor(seq))
        handle.remove()
    except Exception as e:
        handle.remove()
        raise e

    return logits


def neuron_ablated_loss(data, neuron):
    data_size = data.shape[0]
    target = einops.repeat(
        torch.tensor([0.0] * 9 + [1.0]),
        "logit_dim -> data_size logit_dim",
        data_size=data_size,
    ).to("cuda")
    logits = ablate_one_neuron(neuron, data)[:, -1, :]
    loss = F.cross_entropy(logits, target, reduction="none")
    loss.to("cpu").detach().numpy()
    return loss


# Generic wrapper for easily capturing specific things in the model
# Ex. capture_forward_pass(model, model.blocks[0].hook_resid_mid, torch.tensor([10,1,2,3]))
def capture_forward_pass(model, model_path: torch.nn.Module, seq: torch.Tensor):
    def hook(module, input, output):
        captured = output.clone()
        model_path.captured = captured

    try:
        handle = model_path.register_forward_hook(hook)
        with torch.no_grad():
            model(seq)
        activations = model_path.captured
        handle.remove()
    except Exception as e:
        raise e
        handle.remove()

    return activations


# Principal component analysis, returns a dict containing commonly sought information
def pca(data: Tensor, variance_explained: float = 0.95) -> Tensor:
    mean_centered: Tensor = data - data.mean(dim=0)
    cov_matrix = torch.mm(mean_centered.t(), mean_centered) / (
        mean_centered.shape[0] - 1
    )
    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
    real_eigenvalues: Tensor = eigenvalues.float()
    sorted_indices = torch.argsort(real_eigenvalues, descending=True)
    sorted_eigenvalues = real_eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices].float()
    total_variance = torch.sum(sorted_eigenvalues)
    R2 = sorted_eigenvalues / total_variance
    cumulative_explained_variance = torch.cumsum(R2, dim=0)
    components = (
        torch.where(cumulative_explained_variance >= variance_explained)[0][0].item()
        + 1
    )
    k_eigens = sorted_eigenvectors[:, :components]
    projected_data = torch.mm(mean_centered, k_eigens)
    out = {
        "R2": R2[:components],
        "variances": sorted_eigenvalues[:components],
        "principal components": k_eigens,
        "projected data": projected_data,
    }

    return out


# TODO: add options to put the original ordering back
# Use this to get fast inference even with games of differing lengths, if you don't care about the order they come back in
# can also pass in a lambda wrapping capture_forward_pass as well
def inference_on_games(model: HookedTransformer, games: list[list[int]]):
    with torch.no_grad():
        l_games = list(map(lambda l: len(l), games))
        list.sort(l_games)
        lens = list(set(l_games))
        lens.sort()
        # games are ascending, find first of each kind
        batch_indices = [l_games.index(l) for l in lens] + [len(games) - 1]
        data = [
            torch.tensor(games[s:e])
            for s, e in zip(batch_indices[:-1], batch_indices[1:])
        ]
        logits = [model(games) for games in data]
        return logits


def modulate_features(
    autoenc,
    model,
    seq: Tensor,
    feature_modulations: Optional[list[tuple[int, float]]] = None,
    straight_passthrough=False,
) -> Tensor:
    def hook(module, input, output):
        result = output.clone()
        out_w_ablation = autoenc(result, feature_modulations=feature_modulations)[2]
        out = autoenc(result)[2]

        if straight_passthrough:
            return out
        else:
            return out_w_ablation + (result - out)

    try:
        with torch.no_grad():
            handle = model.blocks[0].mlp.hook_post.register_forward_hook(hook)
            logits = model(seq)
            handle.remove()
    except Exception as e:
        handle.remove()
        raise e

    return logits
