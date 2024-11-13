from dash import Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from api import global_view_points
from view.components.component import Component
from view.components.html import described_input
from view.components.style import STYLE_BORDER_INNER


class Case(Component):
    def __init__(self, id: str) -> None:
        super(Case, self).__init__(id, "Load data")
        self.fields = [
            "limit",
            "batch-size",
        ]

    def init_callbacks(self, app, manifold):
        super(Case, self).init_callbacks(app)

        @app.callback(
            Output(*self.data_help),
            Input(*self.submit),
            *[State(*s) for s in self.options()],
            prevent_init_call=True,
        )
        def load_global_points(n_clicks, *options):
            options = self.options_values(options)
            return global_view_points(manifold, options)

    def inner_html(self):
        return html.Div(
            [
                described_input("Examples limit", "number", self.ids(0), 1000),
                described_input("Batch size", "number", self.ids(1), 100),
            ],
            style=STYLE_BORDER_INNER,
        )
