import dash_bootstrap_components as dbc
import torch
from dash import Dash, html
from torchvision import datasets

from models.vae_softplus.vae import ManifoldVae
from view.case import Case
from view.components.html import col, row
from view.components.style import STYLE_BACKGROUND_BACK, STYLE_CENTER, STYLE_VIEW
from view.contour_map import ContourMap
from view.directed_geodesic import ExpmapPaths
from view.global_settings import GlobalSettings
from view.local_settings import LocalSettings
from view.path_info import PathInfo
from view.plot import Plot
from view.point_info import PointInfo
from view.point_selection import PointSelection
from view.shortest_path import ShortestPaths

# CREATE CASE SPECIFIC MANIFOLD
dim = 3
dataset = datasets.MNIST
# dataset = datasets.FashionMNIST
manifold = ManifoldVae(dim, dataset)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global view components
global_view = Plot("global-view", "Latent space")
global_settings = GlobalSettings("global-settings")
case = Case("global-view")
global_point_info = PointInfo("global-point-info", "Point preview")
global_shortest_paths = ShortestPaths("global-shortest-paths")
global_shortest_path_info = PathInfo("global-shortest-path-info", "Shortest paths")

# Local view components
local_view = Plot("local-view", "Latent space around selected point")
local_settings = LocalSettings(
    "reducer", "Local view settings and quality of local dimensionality reduction"
)
selection = PointSelection("selection")
contour_map = ContourMap("contour-map")
local_contour_map_info = PathInfo("local-contour-map-info", "Contour map")
local_point_info = PointInfo("local-point-info", "Point preview")
local_shortest_paths = ShortestPaths("local-shortest-paths")
local_shortest_path_info = PathInfo("local-shortest-path-info", "Shortest paths")
local_expmap_paths = ExpmapPaths("local-expmap-paths")
local_expmap_path_info = PathInfo("local-expmap-path-info", "Directed geodesics")

# init callbacks
global_view.init_callbacks(
    app,
    manifold,
    global_settings,
    case,
    [
        case,
        global_shortest_paths,
    ],
    global_settings,
    local_settings,
)
global_settings.init_callbacks(app, case)
case.init_callbacks(app, manifold)
global_point_info.init_callbacks(app, manifold, global_view)
global_shortest_paths.init_callbacks(app, manifold, global_settings, global_view)
global_shortest_path_info.init_callbacks(app, manifold, global_shortest_paths)

local_view.init_callbacks(
    app,
    manifold,
    local_settings,
    case,
    [
        local_settings,
        contour_map,
        local_shortest_paths,
        local_expmap_paths,
    ],
    global_settings,
    local_settings,
)
local_settings.init_callbacks(app, manifold, selection, case)
selection.init_callbacks(app, manifold, local_settings, global_view, local_view)
contour_map.init_callbacks(app, manifold, local_settings, selection)
local_contour_map_info.init_callbacks(app, manifold, contour_map)
local_point_info.init_callbacks(app, manifold, local_view)
local_shortest_paths.init_callbacks(app, manifold, local_settings, local_view)
local_shortest_path_info.init_callbacks(app, manifold, local_shortest_paths)
local_expmap_paths.init_callbacks(app, manifold, local_settings, selection)
local_expmap_path_info.init_callbacks(app, manifold, local_expmap_paths)

# HTML Layout
global_components = row(
    [
        col(4, case.html()),
        col(4, global_settings.html()),
        col(4, global_point_info.html()),
    ]
)

global_paths = row(
    [
        col(3, global_shortest_paths.html()),
        col(9, global_shortest_path_info.html()),
    ]
)

local_components = row(
    [
        col(4, selection.html()),
        col(4, local_settings.html()),
        col(4, local_point_info.html()),
    ]
)

local_paths = row(
    [
        col(4, contour_map.html()),
        col(8, local_contour_map_info.html()),
        col(4, local_expmap_paths.html()),
        col(8, local_expmap_path_info.html()),
        col(4, local_shortest_paths.html()),
        col(8, local_shortest_path_info.html()),
    ]
)

app.layout = html.Div(
    children=[
        html.H1("Latent space visualisation", style={"textAlign": "center"}),
        html.H6("Global view", style=STYLE_CENTER),
        html.Div(
            [
                global_components,
                global_view.html(),
                global_paths,
            ],
            style=STYLE_VIEW,
        ),
        html.H6("Local view", style=STYLE_CENTER),
        html.Div(
            [
                local_components,
                local_view.html(),
                local_paths,
            ],
            style=STYLE_VIEW,
        ),
    ],
    style=STYLE_BACKGROUND_BACK,
)

if __name__ == "__main__":
    app.run_server(debug=True)
