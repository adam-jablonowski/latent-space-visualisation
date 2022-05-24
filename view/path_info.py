import numpy as np
import torch

from api import path_info
from dash import Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from view.components.figure import Figure
import plotly.express as px


class PathInfo(Figure):

    def __init__(self, id: str, desc: str):
        super(PathInfo, self).__init__(id, desc)
        
    def init_callbacks(self, app, decoder, paths):

        @app.callback(
            Output(*self.graph),
            Input(*paths.data),
            prevent_initial_call=True,
        )
        def submit(points):
            if points is None:
                return self.make_graph(px.scatter())
            else:
                return self.make_graph(path_info(points, decoder))
