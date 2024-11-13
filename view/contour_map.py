import numpy as np
from dash import Input, Output, State, dcc, html

from api import contour_maps_points
from view.components.component import Component
from view.components.html import described_input
from view.components.style import STYLE_BORDER_INNER, STYLE_CENTER


class ContourMap(Component):
    def __init__(self, id: str) -> None:
        super(ContourMap, self).__init__(id, "Contour map")
        self.result = f"{id}-result", "children"
        self.fields = [
            "radius",
            "levels",
            "angle",
            "vector-length",
            "method",
            "first_step",
            "max_step",
            "rtol",
            "atol",
            "lband",
            "uband",
            "min_step",
        ]
        self.bool_fields = [
            "vertical",
            "horizontal",
        ]

    def inner_html(self):
        return html.Div(
            [
                "Choose parameters of contour maps around the point selcted for local view and display the mappings in the local view visualisation.",
                html.Div(
                    [
                        described_input(
                            "Number of vertical contours", "number", self.ids(2), 12
                        ),
                        described_input(
                            "Length of initial direction vectors for vertical contours (float)",
                            "float",
                            self.ids(3),
                            1,
                        ),
                        described_input(
                            "Time of vertical contours curves (float)",
                            "float",
                            self.ids(0),
                            3,
                        ),
                        described_input(
                            "Number of horizontal contours", "number", self.ids(1), 5
                        ),
                        described_input(
                            "Display vertical contours", "bool", self.ids(12), True
                        ),
                        described_input(
                            "Display horizontal contours", "bool", self.ids(13), True
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
        super(ContourMap, self).init_callbacks(app)

        @app.callback(
            Output(*self.data_help),
            Output(*self.result),
            Input(*self.submit),
            State(*settings.data),
            State(*selected.data),
            *[State(*s) for s in self.options()],
            prevent_initial_call=True,
        )
        def submit(n_clicks, settings, selected, *options):
            options = self.options_values(options)
            component = contour_maps_points(manifold, settings, selected, options)
            result = []
            for t in component["sets"]:
                result += [
                    "options:",
                    html.Br(),
                    str(t["info"]["options"]),
                    html.Br(),
                    "stats:",
                    html.Br(),
                    str(t["info"]["stats"]),
                    html.Br(),
                    html.Br(),
                ]
            return component, result
