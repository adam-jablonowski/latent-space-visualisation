import numpy as np
from dash import Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from api import add_shortest_path
from view.components.component import Component
from view.components.html import col, described_input, row, twice
from view.components.style import STYLE_BORDER, STYLE_BORDER_INNER, STYLE_CENTER


class ShortestPaths(Component):
    def __init__(self, id: str) -> None:
        super(ShortestPaths, self).__init__(id, "Shortest paths")
        self.result = f"{id}-result", "children"
        self.fields = [
            "beg-x",
            "beg-y",
            "beg-latent",
            "beg-source",
            "end-x",
            "end-y",
            "end-latent",
            "end-source",
            "levels",
            "max_nodes",
            "tol",
            "T",
        ]
        self.beg = f"{self.id}-beg", "n_clicks"
        self.end = f"{self.id}-end", "n_clicks"
        self.selection_update = f"{self.id}-selection-update", "data"

    def inner_html(self):
        return html.Div(
            [
                dcc.Store(id=self.selection_update[0]),
                "Choose parameters of a shortest path and add the path in the latent space visualization.",
                html.Div(
                    [
                        described_input(
                            "x of the beginning in 2d-reduced dimension (float)",
                            "float",
                            self.ids(0),
                            4,
                        ),
                        described_input(
                            "y of the beginning in 2d-reduced dimension (float)",
                            "float",
                            self.ids(1),
                            0,
                        ),
                        described_input(
                            "latent value of the beginning (Python`s list of floats) ",
                            "text",
                            self.ids(2),
                            "",
                        ),
                        described_input(
                            "input used as the beginning",
                            "dropdown",
                            self.ids(3),
                            [
                                "latent value",
                                "x and y in 2d-reduced dimension (dimesinality reduction reconstruction)",
                            ],
                        ),
                        html.Button(
                            id=self.beg[0],
                            n_clicks=0,
                            children="Set the preview point as the beginning",
                        ),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                html.Div(
                    [
                        described_input(
                            "x of the end in 2d-reduced dimension (float)",
                            "float",
                            self.ids(4),
                            0,
                        ),
                        described_input(
                            "y of the end in 2d-reduced dimension (float)",
                            "float",
                            self.ids(5),
                            2,
                        ),
                        described_input(
                            "latent value of the end (Python`s list of floats) ",
                            "text",
                            self.ids(6),
                            "",
                        ),
                        described_input(
                            "input used as the end",
                            "dropdown",
                            self.ids(7),
                            [
                                "latent value",
                                "x and y in 2d-reduced dimension (dimesinality reduction reconstruction)",
                            ],
                        ),
                        html.Button(
                            id=self.end[0],
                            n_clicks=0,
                            children="Set the preview point as the end",
                        ),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                html.Div(
                    [
                        described_input(
                            "Number of displayed points", "float", self.ids(8), 10
                        ),
                        described_input(
                            "Initial number of path nodes (float)",
                            "float",
                            self.ids(11),
                            30,
                        ),
                        described_input(
                            "Solver`s max number of path nodes",
                            "number",
                            self.ids(9),
                            50,
                        ),
                        described_input(
                            "Target convergence tol (float)", "float", self.ids(10), 1
                        ),
                    ],
                    style=STYLE_BORDER_INNER,
                ),
                html.Div("Solver messages"),
                html.Div([], id=self.result[0], style=STYLE_BORDER_INNER),
            ]
        )

    def init_callbacks(self, app, manifold, settings, plot):
        super(ShortestPaths, self).init_callbacks(app)

        @app.callback(
            Output(*self.data_help),
            Output(*self.result),
            Output(*self.selection_update),
            Input(*self.submit),
            State(*self.result),
            State(*self.data),
            State(*settings.data),
            *[State(*s) for s in self.options()],
            prevent_initial_call=True,
        )
        def submit(n_clicks, result, component, settings, *options):
            options = self.options_values(options)
            component, beg_2d, beg_latent, end_2d, end_latent = add_shortest_path(
                manifold,
                component,
                settings,
                options,
            )
            result += [
                f'Shortest path {len(component["sets"]) // 2}',
                html.Br(),
                "options:",
                str(component["sets"][0]["info"]["options"]),
                html.Br(),
                "stats:",
                str(component["sets"][0]["info"]["stats"]),
                html.Br(),
                html.Br(),
            ]
            return (
                component,
                result,
                (
                    *beg_2d,
                    str(beg_latent.tolist()),
                    *end_2d,
                    str(end_latent.tolist()),
                ),
            )

        @app.callback(
            *[Output(*self.options()[i]) for i in [0, 1, 2, 4, 5, 6]],
            Input(*self.beg),
            Input(*self.end),
            State(*plot.click),
            Input(*self.selection_update),
            *[State(*s) for s in self.options()],
            prevent_initial_call=True,
        )
        def update_beg(
            beg_n_clicks, end_n_clicks, click_data, selection_update, *options
        ):
            options = self.options_values(options)
            triggered_id = ctx.triggered_id
            if triggered_id == self.selection_update[0]:
                return selection_update
            else:
                if click_data is None:
                    raise PreventUpdate
                point = click_data["points"][0]
                if triggered_id == self.beg[0]:
                    return (
                        point["x"],
                        point["y"],
                        point["text"],
                        options["end-x"],
                        options["end-y"],
                        options["end-latent"],
                    )
                elif triggered_id == self.end[0]:
                    return (
                        options["beg-x"],
                        options["beg-y"],
                        options["beg-latent"],
                        point["x"],
                        point["y"],
                        point["text"],
                    )
                else:
                    raise Exception
