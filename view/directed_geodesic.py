from dash import Input, Output, State, html

from api import add_expmap_path
from view.components.component import Component
from view.components.html import described_input
from view.components.style import STYLE_BORDER_INNER


class ExpmapPaths(Component):
    def __init__(self, id: str):
        super(ExpmapPaths, self).__init__(id, "Directed geodesics")
        self.result = f"{id}-result", "children"
        self.fields = [
            "dir-x",
            "dir-y",
            "radius",
            "levels",
            "method",
            "first_step",
            "max_step",
            "rtol",
            "atol",
            "lband",
            "uband",
            "min_step",
        ]

    def inner_html(self):
        return html.Div(
            [
                "Choose parameters of a directed geodesic and add the geodesic in the local view visualisation. ",
                "Starting point is the point selected for the local view",
                html.Div(
                    [
                        described_input(
                            "x of initial direction in 2d-reduced dimension (float)",
                            "float",
                            self.ids(0),
                            1,
                        ),
                        described_input(
                            "y of initial direction in 2d-reduced dimension (float) ",
                            "float",
                            self.ids(1),
                            0,
                        ),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                html.Div(
                    [
                        described_input(
                            "Time of geodesic curve (float)", "float", self.ids(2), 1
                        ),
                        described_input(
                            "Number of displayed points", "number", self.ids(3), 5
                        ),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                html.Div(
                    [
                        described_input(
                            "Method",
                            "dropdown",
                            self.ids(4),
                            [
                                "BDF",
                                "LSODA",
                                "Radau",
                                "DOP853",
                                "RK23",
                                "RK45",
                            ],
                        ),
                        described_input("first-step", "float", self.ids(5), None),
                        described_input("max-step", "float", self.ids(6), None),
                        described_input("rtol", "float", self.ids(7), None),
                        described_input("atol", "float", self.ids(8), None),
                        described_input("lband", "number", self.ids(9), None),
                        described_input("uband", "number", self.ids(10), None),
                        described_input("min-step", "float", self.ids(11), None),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                html.Div("Solver messages"),
                html.Div([], id=self.result[0], style=STYLE_BORDER_INNER),
            ]
        )

    def init_callbacks(self, app, manifold, settings, selected):
        super(ExpmapPaths, self).init_callbacks(app)

        @app.callback(
            Output(*self.data_help),
            Output(*self.result),
            Input(*self.submit),
            State(*self.data),
            State(*self.result),
            State(*settings.data),
            State(*selected.data),
            *[State(*s) for s in self.options()],
            prevent_initial_call=True,
        )
        def submit(n_clicks, component, result, settings, selected, *options):
            options = self.options_values(options)
            component = add_expmap_path(
                manifold,
                settings,
                selected,
                component,
                options,
            )
            result += [
                f'Exponential path {len(component["sets"]) // 2}',
                html.Br(),
                "options:",
                str(component["sets"][0]["info"]["options"]),
                html.Br(),
                "stats:",
                str(component["sets"][0]["info"]["stats"]),
                html.Br(),
                html.Br(),
            ]
            return component, result
