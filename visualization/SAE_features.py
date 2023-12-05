import dash
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
from alphatoe import plot, game, interpretability, models
import plotly.express as px
import pickle
import torch
from functools import partial

with open("./scripts/SAE_features_by_token_8_move_games.pkl", "rb") as f:
    features_by_content = pickle.load(f)

with open("./scripts/8_move_games.pkl", "rb") as f:
    moves_by_content = pickle.load(f)


def to_list(tensor):
    return [i.item for i in tensor]


logits_data = []

autoenc = models.SparseAutoEncoder(512, 512).cuda()
autoenc.load_state_dict(
    torch.load(
        "./notebooks/512_sparse_autoencoder_on_activations_20NOV2023_parameters.pt"
    )
)
model = interpretability.load_model(
    "./scripts/models/prob all 8 layer control-20230718-185339"
)

modulate_features = partial(
    interpretability.modulate_features,
    autoenc,
    model,
)


def cat_moves(content):
    return torch.cat(
        [
            moves_by_content[content]["contains"],
            moves_by_content[content]["not_contains"],
        ],
        dim=0,
    )


def cat_features(content):
    return torch.cat(
        [
            features_by_content[content]["contains"]["idx -1 activations"],
            features_by_content[content]["not_contains"]["idx -1 activations"],
            features_by_content[content]["contains"]["idx -2 activations"],
            features_by_content[content]["not_contains"]["idx -2 activations"],
        ],
        dim=0,
    )


app = Dash(__name__)


app.layout = html.Div(
    [
        html.H1(
            "Sparse Autoencoder Features and Activations by Token",
            style={"textAlign": "center"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Primary Features Across 8000 Games"),
                        html.H3("Select a Token"),
                        dcc.Dropdown(
                            list(features_by_content.keys()),
                            0,
                            id="dropdown-selection-1",
                        ),
                        dcc.Graph(id="graph-content-1"),
                    ],
                    style={
                        "flex": 1,
                        "display": "flex",
                        "flexDirection": "column",
                        "padding": "10px",
                    },
                ),
                html.Div(
                    [
                        html.H2("Activations of Selected Feature for 8000 Games"),
                        html.H3("Select a Feature"),
                        dcc.Dropdown(id="dropdown-selection-2"),
                        dcc.Graph(id="graph-content-2"),
                    ],
                    style={
                        "flex": 1,
                        "display": "flex",
                        "flexDirection": "column",
                        "padding": "10px",
                    },
                ),
            ],
            style={"display": "flex", "flexDirection": "row", "gap": "20px"},
        ),
        html.Pre("Click on a bar to see the value!", id="some-output"),
        dcc.Graph(id="logits-graph"),
        dcc.Store(id="filtered-indices"),
    ],
    style={"padding": "20px"},
)


@callback(
    Output("graph-content-1", "figure"),
    Output("dropdown-selection-2", "options"),
    Output("dropdown-selection-2", "value"),
    Output("filtered-indices", "data"),
    Input("dropdown-selection-1", "value"),
)
def update_graph_1_and_dropdown(value):
    catted_features = cat_features(value)
    figure = plot.imshow_comp_acts(
        catted_features,
        groups=[
            "GameOver - Token Present",
            "GameOver, Token Not Present",
            "Game Ongoing, Token Present",
            "Game Ongoing, Token Not Present",
        ],
        show_fig=False,
    )
    options = plot.get_filtered_acts(catted_features.T, just_indices=True)
    default_value = options[0] if options else None
    return figure, options, default_value, options


@callback(
    Output("graph-content-2", "figure"),
    Input("dropdown-selection-1", "value"),
    Input("dropdown-selection-2", "value"),
)
def update_graph_2(value1, value2):
    catted_features = cat_features(value1)

    secondary_figure = plot.hist_activations(
        catted_features,
        value2,
        [
            "GameOver - Token Present",
            "GameOver, Token Not Present",
            "Game Ongoing, Token Present",
            "Game Ongoing, Token Not Present",
        ],
    )
    return secondary_figure


def create_logits_heatmap(logits_matrix):
    # Create the heatmap
    print(logits_matrix)
    fig = px.imshow(
        logits_matrix,
        labels=dict(x="Logit Index", y="Logit Type", color="Logit Value"),
        x=["MLP", "Normal", "Ablated", "Intensified"],
        color_continuous_scale="RdBu",  # Red-Blue diverging color scale
    )
    fig.update_layout(title="Logits Comparison")

    # Annotate each cell with its value
    for y in range(logits_matrix.shape[0]):
        for x in range(logits_matrix.shape[1]):
            fig.add_annotation(
                x=x,
                y=y,
                text=str(
                    round(logits_matrix[y, x].item(), 5)
                ),  # Convert tensor value to string
                showarrow=False,
                font=dict(
                    color="black" if abs(logits_matrix[y, x].item()) < 0.5 else "white"
                ),
            )

    # Update color scale to center at 0
    fig.update_traces(
        colorbar=dict(tickvals=[logits_matrix.min(), 0, logits_matrix.max()])
    )
    return fig


def prepare_logits_for_heatmap(
    mlp_logits, normal_logits, ablated_logits, intensified_logits
):
    logits_matrix = (
        torch.stack([mlp_logits, normal_logits, ablated_logits, intensified_logits])
        .cpu()
        .T
    )
    print(logits_matrix)
    return logits_matrix


@app.callback(
    Output(
        "some-output", "children"
    ),  # Ensure this Output is suitable for the data type you're returning
    Output("logits-graph", "figure"),
    Input(
        "graph-content-1", "clickData"
    ),  # Listening to click events on the first graph
    State(
        "dropdown-selection-1", "value"
    ),  # Listening to the value of the first dropdown
    State(
        "dropdown-selection-2", "value"
    ),  # Listening to the value of the second dropdown
    State("filtered-indices", "data"),
)
def display_click_data(clickData, dropdown_value1, dropdown_value2, filtered_indices):
    if clickData:
        x_value = clickData["points"][0]["x"]
        y_value = clickData["points"][0]["y"]
        feature_index = filtered_indices[int(y_value)]
        activation = clickData["points"][0]["z"]
        moves = cat_moves(dropdown_value1)
        seq = list(moves[x_value % 4000])

        if x_value // 4000 > 0:
            seq.pop()

        tseq = torch.tensor([10] + seq)
        print(f"y_value is {y_value}")
        with torch.no_grad():
            mlp_logits = model(tseq)[0, -1, :]
            normal_logits = modulate_features(tseq)[0, -1, :]
            ablated_logits = modulate_features(tseq, [(int(feature_index), 0)])[
                0, -1, :
            ]
            intensified_logits = modulate_features(tseq, [(int(feature_index), 2)])[
                0, -1, :
            ]
        logits_matrix = prepare_logits_for_heatmap(
            mlp_logits, normal_logits, ablated_logits, intensified_logits
        )
        logits_heatmap = create_logits_heatmap(logits_matrix)
        # Format the logits and other information for display
        formatted_seq = " ".join([str(c.item()) for c in seq])
        board_sketch = game.play_game(seq).sketch_board()
        # logits_info = f"MLP Logits: {mlp_logits}\nNormal Logits: {normal_logits}\nAblated Logits: {ablated_logits}\nIntensified Logits: {intensified_logits}"

        return (
            f"{formatted_seq}\n{board_sketch}\nActivation: {activation}\nFeature Index: {feature_index}",
            logits_heatmap,
        )

    return "Click on a bar to see the value!", dash.no_update


if __name__ == "__main__":
    app.run(debug=True)
