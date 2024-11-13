import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html

from api import select_point
from view.components.component import Component
from view.components.html import described_input
from view.components.style import STYLE_BORDER_INNER, STYLE_CENTER


class PointSelection(Component):
    def __init__(self, id: str) -> None:
        super(PointSelection, self).__init__(id, "Selected point for local view")
        self.figure = f"{id}-figure", "figure"
        self.fields = [
            "x",
            "y",
            "latent",
            "source",
        ]

    def inner_html(self):
        return html.Div(
            [
                html.Div("Select point for local view", style=STYLE_CENTER),
                html.Div(
                    [
                        described_input("Clicked x (float)", "float", self.ids(0), 0),
                        described_input("Clicked y (float) ", "float", self.ids(1), 0),
                        described_input(
                            "Latent point (float list)", "text", self.ids(2), ""
                        ),
                        described_input(
                            "Method",
                            "dropdown",
                            self.ids(3),
                            [
                                "global view",
                                "local view",
                                "latent value",
                                "reconstructed from xy",
                            ],
                        ),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                dcc.Graph(
                    id=self.figure[0],
                    figure=px.scatter(height=300),
                ),
            ]
        )

    def init_callbacks(
        self,
        app,
        manifold,
        settings,
        global_plot,
        local_plot,
    ):
        super(PointSelection, self).init_callbacks(app)

        @app.callback(
            Output(*self.data_help),
            Output(*self.figure),
            Output(*self.options()[0]),
            Output(*self.options()[1]),
            Output(*self.options()[2]),
            Input(*self.submit),
            Input(*self.reset),
            State(*settings.data),
            State(*global_plot.click),
            State(*local_plot.click),
            *[State(*s) for s in self.options()],
            prevent_initial_call=True,
        )
        def select(
            n_clicks,
            n_clicks_reset,
            settings,
            global_click,
            local_click,
            *options,
        ):
            triggered_id = ctx.triggered_id
            if triggered_id == self.submit[0]:
                options = self.options_values(options)
                latent, point_2d = select_point(
                    settings, global_click, local_click, options
                )
                figure = go.Figure(manifold.point_info(latent))
                return latent, figure, *point_2d, str(latent.tolist())
            elif triggered_id == self.reset[0]:
                print("yolo")
                return None, px.scatter(), 0, 0, ""
