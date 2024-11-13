import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch

from neighbours import Neighbours
from reducer import Reducer


class PlotFactory:
    def __init__(
        self,
        settings,
        case,
        manifold,
        global_reduction,
        local_reduction,
    ):
        self.settings = settings
        self.reducer = Reducer(repr=self.settings["projector"])
        self.case = case
        self.neighbors = Neighbours(repr=case["neighbours"])
        self.manifold = manifold
        self.global_reducer = Reducer(repr=global_reduction["projector"])
        if local_reduction:
            self.local_reducer = Reducer(repr=local_reduction["projector"])

    def add_cmin_cmax_of_components(self, components):
        settings = self.settings
        for d in components:
            if d and d["sets"]:
                for t in d["sets"]:
                    _min, _max = t["info"]["minmeasure"], t["info"]["maxmeasure"]
                    if "minmeasure" not in settings:
                        settings["minmeasure"] = _min
                    if "maxmeasure" not in settings:
                        settings["maxmeasure"] = _max
                    settings["minmeasure"] = min(settings["minmeasure"], _min)
                    settings["maxmeasure"] = max(settings["minmeasure"], _max)

    def make_figure(self, components, height=700):
        self.add_cmin_cmax_of_components(components)
        fig = go.Figure(layout=go.Layout(height=height))
        for d in components:
            if d and d["sets"]:
                for t in d["sets"]:
                    self.add_trace(fig, t)
        return fig

    def add_trace(self, fig, trace):
        reducer = Reducer(repr=self.settings["projector"])
        projected = self.reducer.reduce(trace["values"]["latent"])

        z = self.make_axis(trace, self.settings["z-axis"], self.settings["log-height"])
        color = self.make_axis(
            trace, self.settings["color"], self.settings["log-color"]
        )
        markers = self.markers(trace, color, size=5 if z is None else 2)
        text = [str(l) for l in trace["values"]["latent"]]
        target = self.arrows(trace)

        if z is None:
            fig.add_trace(
                go.Scatter(
                    x=projected[:, 0],
                    y=projected[:, 1],
                    text=text,
                    **markers,
                )
            )
            if target is not None:
                quiver = ff.create_quiver(
                    x=projected[:, 0],
                    y=projected[:, 1],
                    u=target[:, 0] - projected[:, 0],
                    v=target[:, 1] - projected[:, 1],
                    text=text,
                )
                quiver.update_traces(line_color="gray")
                fig.add_traces(data=quiver.data)
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=projected[:, 0],
                    y=projected[:, 1],
                    text=text,
                    z=z,
                    **markers,
                )
            )
            if target is not None:
                fig.add_trace(
                    go.Cone(
                        x=projected[:, 0],
                        y=projected[:, 1],
                        z=z,
                        u=target[:, 0] - projected[:, 0],
                        v=target[:, 1] - projected[:, 1],
                        w=np.zeros_like(z),
                        text=text,
                    )
                )

    def make_axis(self, trace, axis_type, log_values):
        values = self.axis_values(trace, axis_type)
        return np.log(values) if log_values and (values is not None) else values

    def axis_values(self, trace, axis_type):
        if axis_type == "none":
            return None
        elif axis_type == "label":
            labels = trace["values"]["label"]
            if labels is not None:
                return labels
            else:
                return [-1] * len(trace["values"]["latent"])
        else:
            latent = trace["values"]["latent"]
            idxs = self.neighbors.neighbours_idxs(latent, self.settings["me-radius"])

            if axis_type == "measure":
                field_name = "measure"
                if_none = lambda x: self.manifold.measure(np.array(x))
                value = lambda x: x
            elif axis_type == "encoder + decoder":
                field_name = "loss"
                if_none = None
                value = lambda x: x
            elif axis_type == "global reduction":
                field_name = "latent"
                if_none = lambda x: np.expand_dims(x, axis=0)
                value = lambda x: self.global_reducer.reconstruction_quality(x)
            elif axis_type == "encoder + global reduction + decoder":
                field_name = "observation"
                if_none = None

                def value(observation):
                    mu, log_var = self.manifold.encode(observation)
                    mu = self.global_reducer.reduce(mu)
                    mu = self.global_reducer.reconstruct(mu)
                    recon = self.manifold.decode(mu)
                    return self.manifold.loss_function(recon, observation, mu, log_var)

            elif axis_type == "local reduction":
                field_name = "latent"
                if_none = lambda x: np.expand_dims(x, axis=0)
                value = lambda x: self.local_reducer.reconstruction_quality(x)
            elif axis_type == "encoder + local reduction + decoder":
                field_name = "observation"
                if_none = None

                def value(observation):
                    mu, log_var = self.manifold.encode(observation)
                    mu = self.local_reducer.reduce(mu)
                    mu = self.local_reducer.reconstruct(mu)
                    recon = self.manifold.decode(mu)
                    return self.manifold.loss_function(recon, observation, mu, log_var)

            else:
                raise NotImplementedError("Unknown axis type")

            cached = np.concatenate(
                [t["values"][field_name] for t in self.case["sets"]], axis=0
            )
            neighbours = []
            for ii, i in enumerate(idxs):
                if len(i) == 0:
                    if if_none:
                        neighbours.append(if_none(latent[ii]))
                    else:
                        neighbours.append(None)
                else:
                    neighbours.append(cached[i])
            values = [np.mean(value(n)) if n is not None else 0 for n in neighbours]
            return values

    def markers(self, trace, color, size):
        marker = trace["settings"]["marker"]
        marker["color"] = color
        marker["size"] = size
        if self.settings["color"] == "measure":
            max_min = [self.settings["maxmeasure"], self.settings["minmeasure"]]
            marker["cmax"], marker["cmin"] = (
                np.log(max_min) if self.settings["log-color"] else max_min
            )
        return trace["settings"]

    def arrows(self, trace):
        if self.settings["arrows"] == "none":
            return None
        elif self.settings["arrows"] == "-> decoder + encoder + reduction":
            decoded = self.manifold.decode(torch.Tensor(trace["values"]["latent"]))
            latent, log_var = self.manifold.encode(decoded)
        elif self.settings["arrows"] == "-> reconstruction + reduction":
            reduced = self.reducer.reduce(trace["values"]["latent"])
            latent = self.reducer.reconstruct(reduced)
        elif (
            self.settings["arrows"]
            == "-> reconstruction + decoder + encoder + reduction"
        ):
            reduced = self.reducer.reduce(trace["values"]["latent"])
            latent = self.reducer.reconstruct(reduced)
            decoded = self.manifold.decode(torch.Tensor(latent))
            latent, log_var = self.manifold.encode(decoded)
        else:
            raise NotImplementedError("Unknown arrows targets")
        return self.reducer.reduce(latent)
