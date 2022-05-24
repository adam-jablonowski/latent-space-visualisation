from dash import dcc, html
import plotly.express as px
from view.components.style import STYLE_CENTER

class Figure:

    def __init__(self, id: str, desc: str):
        self.id = id
        self.desc = desc
        self.graph = f'{id}-graph', 'children'
        self.figure = f'{id}-figure', 'figure'
        self.click = f'{id}-figure', 'clickData'
    
    def make_graph(self, figure):
        if isinstance(figure, list):
            return [f if isinstance(f, str) else dcc.Graph(figure=f) for f in figure]
        else:
            return dcc.Graph(
                id=self.figure[0],
                figure=figure,
            )

    def html(self):
        return html.Div([
            html.Div(self.desc, style=STYLE_CENTER),
            html.Div(
                self.make_graph(px.scatter()),
                id=self.graph[0],
            )
        ])