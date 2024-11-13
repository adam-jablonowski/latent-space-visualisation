import numpy as np
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from api import local_dimensionality_reduction
from view.components.component import Component
from view.components.html import arrows_input, axis_input, described_input, twice
from view.components.style import STYLE_BORDER_INNER


class LocalSettings(Component):
    def __init__(self, id: str, desc: str):
        super(LocalSettings, self).__init__(id, desc)
        self.result = f"{id}-result", "children"
        self.fields = [
            "radius",
            "me-radius",
            "color",
            "z-axis",
            "arrows",
        ]
        self.bool_fields = [
            "log-color",
            "log-height",
        ]

    def init_callbacks(self, app, manifold, selection, case):
        super(LocalSettings, self).init_callbacks(app)

        @app.callback(
            Output(*self.data_help),
            Output(*self.result),
            Input(*self.submit),
            Input(*selection.data),
            State(*case.data),
            *[State(*s) for s in self.options()],
            prevent_initial_call=True,
        )
        def quality(n_clicks, selected, case, *options):
            if selected is None:
                raise PreventUpdate
            options = self.options_values(options)
            return local_dimensionality_reduction(manifold, selected, case, options)

    def inner_html(self):
        return html.Div(
            [
                "Choose settings of the visulisation. ",
                "Choose the radius of the visualised neighbourhood of the selected point and compute the quality of dimensionality reduction. ",
                "The component stores dimensionality reduction parameters.",
                html.Div(
                    [
                        described_input(
                            "Radius for neighbourhood", "number", self.ids(0), 1
                        ),
                        described_input(
                            "Radius for mean error", "float", self.ids(1), 0.1
                        ),
                        *axis_input(
                            "Plot`s color",
                            self.ids(2),
                            "Turn logarithm color on",
                            self.ids(5),
                        ),
                        *axis_input(
                            "Plot`s z-axis",
                            self.ids(3),
                            "Turn logarithm z-axis on",
                            self.ids(6),
                        ),
                        arrows_input(self.ids(4)),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                "Mean encoder + local dim reduction + decoder quality",
                html.Div(" " * 10, id=self.result[0], style=STYLE_BORDER_INNER),
            ]
        )
