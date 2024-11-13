from dash import Input, Output, State, ctx, html
from dash.exceptions import PreventUpdate

from api import global_dimensionality_reducion
from view.components.component import Component
from view.components.html import arrows_input, axis_input, described_input
from view.components.style import STYLE_BORDER_INNER


class GlobalSettings(Component):
    def __init__(self, id: str) -> None:
        super(GlobalSettings, self).__init__(id, "Global view settings")
        self.fields = [
            "me-radius",
            "color",
            "z-axis",
            "arrows",
        ]
        self.bool_fields = [
            "log-color",
            "log-height",
        ]

    def init_callbacks(self, app, component):
        super(GlobalSettings, self).init_callbacks(app)

        @app.callback(
            Output(*self.data_help),
            Input(*component.data),
            Input(*self.submit),
            *[State(*s) for s in self.options()],
            prevent_initial_call=True,
        )
        def dimensionality_reducion(component, n_clicks, *options):
            if component is None:
                raise PreventUpdate
            options = self.options_values(options)
            return global_dimensionality_reducion(component, options)

    def inner_html(self):
        return html.Div(
            [
                "Choose settings of the visualisation. ",
                "The component stores dimensionality reduction parameters.",
                html.Div(
                    [
                        described_input(
                            "Radius for mean error", "float", self.ids(0), 0
                        ),
                        *axis_input(
                            "Plot`s color",
                            self.ids(1),
                            "Turn logarithm color on",
                            self.ids(4),
                        ),
                        *axis_input(
                            "Plot`s z-axis",
                            self.ids(2),
                            "Turn logarithm z-axis on",
                            self.ids(5),
                        ),
                        arrows_input(self.ids(3)),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
            ]
        )
