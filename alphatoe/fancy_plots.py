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
    

def number_to_grid_position(num: int) -> tuple:
    if num < 0 or num > 8:
        raise ValueError("Number must be between 0 and 8")
    row = num // 3
    col = num % 3
    return row, col

def map_attn_to_board(attn: Tensor, moves: list[int]) -> np.ndarray:
    attn = to_numpy(attn, flat=True)
    #zip moves and attn together
    moves_attn = list(zip(moves, attn))
    board_attn = np.zeros(9)
    for pair in moves_attn:
        board_attn[pair[0]] = pair[1]
    #reshape board_attn to 3x3
    board_attn = board_attn.reshape(3,3)
    return board_attn

def moves_seq_to_player_moves(moves: list[int]) -> dict:
    board_moves = {
        'x': [],
        'o': []
    }
    for i, move in enumerate(moves):
        if i % 2 == 0:
            board_moves['x'].append(move)
        else:
            board_moves['o'].append(move)
    return board_moves

def add_board_moves(fig, board_moves: dict) -> None:
    annotations = []
    for player, moves in board_moves.items():
        for move in moves:
            row, col = number_to_grid_position(move)
            annotations.append(
                dict(
                    x=col,
                    y=row-0.05,
                    xref="x",
                    yref="y",
                    text=player,
                    showarrow=False,
                    font=dict(size=80, color="#808080", family='Arial'),
                    yanchor="middle",
                    xanchor="center"
                )
            )
    fig.update_layout(annotations=annotations)

def board_heatmap(pattern: Tensor, moves: list, title: str, color: str) -> None:
    pattern = to_numpy(pattern)
    
    game_over_pred = round(float(pattern[-1]), 2)
    #remove last position from pattern
    pattern = pattern[:-1]
    #check that pattern is length 9
    assert len(pattern) == 9, "Pattern must be of length 9"
    #turn pattern into 3x3 matrix
    pattern = pattern.reshape(3,3)

    board_moves = moves_seq_to_player_moves(moves)

    color_scale = [
        [0, '#F5F5F5'],
        [1, color]
    ]

    fig = go.Figure(data=go.Heatmap(
        z = to_numpy(pattern),
        colorscale=color_scale,
        zmin=0,
        zmax=0.5,
        colorbar=dict(
            thickness=15,
            len=1,
            outlinecolor="black",
            outlinewidth=0.5,
            titleside="right"
        )
    ))
    add_board_moves(fig, board_moves)

    #add annotation for game over prediction beneath board
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"Game Over Prediction: {game_over_pred}",
        showarrow=False,
        font=dict(size=25, color="#333333", family='Arial'),
    )

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=30, color="#333333", family='Arial')
        },
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            range=[-0.5, 2.5],
            scaleanchor='y',
            scaleratio = 1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            range=[2.5, -0.5]  # Reverse y-axis to match game board orientation
        ),
        width=500,
        height=500,
        margin=dict(l=40, r=40, t=60, b=60),
        plot_bgcolor='#F5F5F5'
    )
    # Add grid lines and border
    for i in range(4):
        fig.add_shape(type="line", x0=i-0.5, y0=-0.5, x1=i-0.5, y1=2.5, 
                      line=dict(color="black", width=1 if i in [0, 3] else 0.5))
        fig.add_shape(type="line", x0=-0.5, y0=i-0.5, x1=2.5, y1=i-0.5, 
                      line=dict(color="black", width=1 if i in [0, 3] else 0.5))
    #show figure
    fig.show()

def board_attn_heatmap(pattern: Tensor, moves: list, title: str, color: str) -> None:
    #drop first element of pattern
    pattern = pattern[1:]

    attn_board = map_attn_to_board(pattern, moves)

    board_moves = moves_seq_to_player_moves(moves)

    color_scale = [
        [0, '#F5F5F5'],
        [1, color]
    ]

    fig = go.Figure(data=go.Heatmap(
        z = attn_board,
        colorscale=color_scale,
        zmin=0,
        zmax=0.5,
        colorbar=dict(
            thickness=15,
            len=1,
            outlinecolor="black",
            outlinewidth=0.5,
            titleside="right"
        )
    ))
    add_board_moves(fig, board_moves)

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=30, color="#333333", family='Arial')
        },
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            range=[-0.5, 2.5],
            scaleanchor='y',
            scaleratio = 1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            range=[2.5, -0.5]  # Reverse y-axis to match game board orientation
        ),
        width=500,
        height=500,
        margin=dict(l=40, r=40, t=60, b=60),
        plot_bgcolor='#F5F5F5'
    )
    # Add grid lines and border
    for i in range(4):
        fig.add_shape(type="line", x0=i-0.5, y0=-0.5, x1=i-0.5, y1=2.5, 
                      line=dict(color="black", width=1 if i in [0, 3] else 0.5))
        fig.add_shape(type="line", x0=-0.5, y0=i-0.5, x1=2.5, y1=i-0.5, 
                      line=dict(color="black", width=1 if i in [0, 3] else 0.5))
    #show figure
    fig.show()