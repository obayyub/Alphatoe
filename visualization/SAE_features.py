from dash import Dash, html, dcc, callback, Output, Input
from alphatoe import plot
import plotly.express as px
import pickle
import torch

with open("./scripts/SAE_features_by_token_8_move_games.pkl", "rb") as f:
    features_by_content = pickle.load(f)

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
        html.H1("Sparse Autoencoder Features and Acitvations by Token", style={"textAlign": "center"}),
        html.Div(
            [
                html.Div(
                    [
                        html.H2("Primary Features Across 8000 Games"),
                        html.H3("Select a Token"),
                        dcc.Dropdown(list(features_by_content.keys()), 0, id='dropdown-selection-1'),
                        dcc.Graph(id='graph-content-1')
                    ],
                    style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'padding': '10px'}
                ),
                html.Div(
                    [
                        html.H2("Activations of Selected Feature for 8000 Games"),
                        html.H3("Select a Feature"),
                        dcc.Dropdown(id='dropdown-selection-2'),
                        dcc.Graph(id='graph-content-2')
                    ],
                    style={'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'padding': '10px'}
                )
            ],
            style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px'}
        )
    ],
    style={'padding': '20px'}
)


@callback(
    Output('graph-content-1', 'figure'),
    Output('dropdown-selection-2', 'options'),
    Output('dropdown-selection-2', 'value'),
    Input('dropdown-selection-1', 'value')
)
def update_graph_1_and_dropdown(value):
    catted_features = cat_features(value)
    figure =  plot.imshow_comp_acts(
        catted_features,
        groups=["GameOver - Token Present", "GameOver, Token Not Present", "Game Ongoing, Token Present", "Game Ongoing, Token Not Present"],
        show_fig=False,
    )
    options = plot.get_filtered_acts(catted_features.T, just_indices=True)
    default_value = options[0] if options else None
    return figure, options, default_value

@callback(
    Output('graph-content-2', 'figure'),
    Input('dropdown-selection-1', 'value'),
    Input('dropdown-selection-2', 'value')
)
def update_graph_2(value1, value2):
    catted_features = cat_features(value1)

    secondary_figure = plot.hist_activations(catted_features, value2, ["GameOver - Token Present", "GameOver, Token Not Present", "Game Ongoing, Token Present", "Game Ongoing, Token Not Present"])
    return secondary_figure



if __name__ == "__main__":
    app.run(debug=True)
