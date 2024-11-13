import plotly.express as px
from dash import Input, Output, State

from plot import PlotFactory
from view.components.figure import Figure


class Plot(Figure):
    def __init__(self, id: str, desc: str):
        super(Plot, self).__init__(id, desc)

    def init_callbacks(
        self,
        app,
        manifold,
        settings,
        case,
        components,
        global_reduction,
        local_reduction,
    ):
        @app.callback(
            Output(*self.figure),
            Input(*settings.data),
            Input(*case.data),
            State(*global_reduction.data),
            State(*local_reduction.data),
            *[Input(*c.data) for c in components],
        )
        def update(settings, case, global_reduction, local_reduction, *components_data):
            if settings is None:
                return px.scatter()
            else:
                factory = PlotFactory(
                    settings, case, manifold, global_reduction, local_reduction
                )
                return factory.make_figure(components_data, height=600)
