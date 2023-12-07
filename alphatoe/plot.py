import torch
from torch import Tensor
import numpy as np
from functools import partial
from plotly import express as px
from plotly.subplots import make_subplots
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
    show_fig: bool = True,
    animation_name: str = "Snapshot",
    aspect="auto",
    **kwargs: Any
):
    tensor = torch.squeeze(tensor)
    fig = px.imshow(
            to_numpy(tensor, flat=False),
            aspect=aspect,
            labels={"x": xaxis, "y": yaxis, "animation_name": animation_name},
            **kwargs
        )
    if show_fig:
        fig.show()
    else:
        return fig


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
    imshow, color_continuous_scale="RdBu", color_continuous_midpoint=0.0, show_fig=True
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

def get_filtered_acts(acts: Tensor, just_indices: bool = False) -> Tensor:
    zero_elements = acts == 0
    fraction_zeros = torch.mean(zero_elements.float(), dim=1)
    mask = fraction_zeros > 0.90
    inverted_mask = ~mask

    filtered_acts = acts[inverted_mask]

    indices = torch.nonzero(inverted_mask, as_tuple=False).squeeze()

    string_indices = [str(index.item()) for index in indices]


    if just_indices:
        return string_indices
    else:
        return filtered_acts, string_indices

def _get_divisions(groups: list, num_games: int) -> (list, list):
    divisions = []
    #modulus assert
    assert num_games % len(groups) == 0, "number of games must be divisible by number of groups"
    interval = num_games // len(groups)
    for i in range(len(groups)):
        divisions.append(interval*(i+1))
    #divide divisions by 2
    label_ticks = [x-(interval//2) for x in divisions]
    #pop last element off divisons as we don't want a line at the end
    divisions.pop()
    return divisions, label_ticks

def imshow_comp_acts(
    acts: Tensor,
    groups: list,
    line_color: str = 'red',
    show_fig: bool = True,
    ):

    acts = acts.T
    #get number of games
    num_games = acts.shape[1]

    filtered_acts, string_indices = get_filtered_acts(acts)
    fig = imshow_div(filtered_acts, show_fig=False, aspect="auto", width=1000, height=650)

    vert_lines, x_labels = _get_divisions(groups, num_games)

    for line in vert_lines:
        fig.add_vline(x=line, line_width=1, line_dash="dash", line_color=line_color)

    #x label as specific positions
    fig.update_xaxes(
        tickmode = 'array',
        tickvals = x_labels,
        ticktext = groups
    )

    fig.update_yaxes(
        tickmode = 'array',
        tickvals = np.arange(len(string_indices)),
        ticktext = string_indices
    )
    #y axis title
    fig.update_layout(yaxis_title="Feature Index")

    if show_fig:
        fig.show()
    else:
        return fig

def hist_activations(acts: Tensor, feature: int, groups: list):
    acts = to_numpy(acts)
    feature = int(feature)
    fig1 = px.histogram(acts[:2000, feature], nbins=10000)
    fig2 = px.histogram(acts[2000:4000, feature], nbins=100)
    fig3 = px.histogram(acts[4000:6000, feature], nbins=100)
    fig4 = px.histogram(acts[6000:, feature], nbins=10000)


    fig = make_subplots(rows=2, cols=2, subplot_titles=groups)

    for i, f in enumerate([fig1, fig2, fig3, fig4]):
        row = i // 2 + 1  # Determine the row: 1 for the first two, 2 for the next two
        col = i % 2 + 1   # Determine the column: alternates between 1 and 2
        for trace in f.data:
            fig.add_trace(trace, row=row, col=col)

    # Update layout if needed
    fig.update_layout(height=600, width=800)
    #no legend
    fig.update_layout(showlegend=False)
    
    return fig