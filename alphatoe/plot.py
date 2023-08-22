import torch
from torch import Tensor
import numpy as np
from functools import partial
from plotly import express as px
from plotly import graph_objects as go
from typing import Optional, Any


def to_numpy(tensor: Tensor, flat: bool = False) -> Tensor:
    if type(tensor) != torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def imshow(
    tensor: Tensor,
    xaxis: Optional[str] = None,
    yaxis: Optional[str] = None,
    animation_name: str = "Snapshot",
    aspect="auto",
    **kwargs: Any
):
    tensor = torch.squeeze(tensor)
    px.imshow(
        to_numpy(tensor, flat=False),
        aspect=aspect,
        labels={"x": xaxis, "y": yaxis, "animation_name": animation_name},
        **kwargs
    ).show()


def line(x, y=None, hover=None, xaxis="", yaxis="", **kwargs):
    if type(y) == torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x) == torch.Tensor:
        x = to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()


def lines(
    lines_list,
    x=None,
    mode="lines",
    labels=None,
    xaxis="",
    yaxis="",
    title="",
    log_y=False,
    hover=None,
    **kwargs
):
    if type(lines_list) == torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x = np.arange(len(lines_list[0]))
    fig = go.Figure(layout={"title": title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line) == torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(
            go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs)
        )
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()


# Set default colour scheme
imshow = partial(imshow, color_continuous_scale="RdBu")
# Creates good defaults for showing divergent colour scales (ie with both
# positive and negative values, where 0 is white)
imshow_div = partial(
    imshow, color_continuous_scale="RdBu", color_continuous_midpoint=0.0
)
# Presets a bunch of defaults to imshow to make it suitable for showing heatmaps
# of activations with x axis being input 1 and y axis being input 2.
inputs_heatmap = partial(
    imshow,
    xaxis="Input 1",
    yaxis="Input 2",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)
