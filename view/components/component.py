import json

from dash import Input, Output, State, ctx, dcc, html

from view.components.html import button, col, row
from view.components.style import STYLE_COMPONENT


class Component:
    def __init__(self, id: str, desc: str):
        self.id = id
        self.desc = desc
        self.data = f"{id}-data", "data"
        self.data_help = f"{id}-data-help", "data"
        self.upload_contents = f"{id}-upload", "contents"
        self.download_submit = f"{id}-download-submit", "n_clicks"
        self.download = f"{id}-download", "data"
        self.submit = f"{self.id}-submit", "n_clicks"
        self.reset = f"{self.id}-reset", "n_clicks"
        self.fields = []
        self.bool_fields = []

    def options(self):
        return [(f"{self.id}-{f}", "value") for f in self.fields] + [
            (f"{self.id}-{f}", "on") for f in self.bool_fields
        ]

    def options_values(self, values):
        return {f: v for f, v in zip(self.fields + self.bool_fields, values)}

    def ids(self, idx: int):
        return self.options()[idx][0]

    def init_callbacks(self, app):
        @app.callback(
            Output(*self.download),
            Input(*self.download_submit),
            State(*self.data),
            prevent_initial_call=True,
        )
        def download(n_clicks, data):
            return dict(content=json.dumps(data), filename="file.txt")

        @app.callback(
            Output(*self.data),
            Input(*self.upload_contents),
            Input(*self.data_help),
            Input(*self.reset),
        )
        def update(contents, help_data, reset):
            triggered_id = ctx.triggered_id
            if triggered_id == self.data_help[0]:
                return help_data
            elif triggered_id == self.upload_contents[0]:
                return contents
            elif triggered_id == self.reset[0]:
                return None

    def html(self):
        return html.Div(
            [
                dcc.Store(id=self.data[0]),
                dcc.Store(id=self.data_help[0]),
                dcc.Download(
                    id=self.download[0],
                ),
                row(
                    [
                        col(10, html.Div(self.desc, style={"font-size": "20px"})),
                        col(
                            2,
                            html.Div(
                                [
                                    col(4, button(self.download_submit[0], "Download")),
                                    col(
                                        4,
                                        dcc.Upload(
                                            id=self.upload_contents[0],
                                            children=button("", "Upload"),
                                        ),
                                    ),
                                    col(4, button(self.reset[0], "Reset")),
                                ],
                                style={"font-size": "10px"},
                            ),
                        ),
                    ]
                ),
                html.Div(
                    [
                        self.inner_html(),
                        html.Button(id=self.submit[0], n_clicks=0, children="Submit"),
                    ],
                    style={"font-size": "14px"},
                ),
            ],
            style=STYLE_COMPONENT,
        )
