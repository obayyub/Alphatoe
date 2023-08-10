import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import json
import matplotlib.pyplot as plt


def numpy(t):
    return t.cpu().detach().numpy()


def load_model(model_file_name: str):
    weights = torch.load(model_file_name + ".pt")
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
        device=args["device"],
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
