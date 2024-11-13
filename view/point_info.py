from dash import Input, Output
from dash.exceptions import PreventUpdate

from api import point_info
from view.components.figure import Figure


class PointInfo(Figure):
    def __init__(self, id: str, desc: str):
        super(PointInfo, self).__init__(id, desc)

    # TODO original not decoder
    def init_callbacks(self, app, manifold, view):
        @app.callback(
            Output(*self.graph),
            Input(*view.click),
            prevent_initial_call=True,
        )
        def submit(click_data):
            if click_data is None:
                raise PreventUpdate
            return self.make_graph(point_info(click_data, manifold))
