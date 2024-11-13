import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import Data
from equations import (
    all_directions,
    euclidean_direction,
    euclidean_shortest_path,
    solve_expmap,
    solve_shortest_path,
)
from neighbours import Neighbours
from reducer import Reducer


### Utils
def click_data_to_point_2d(click_data):
    return np.array(
        [
            click_data["points"][0]["x"],
            click_data["points"][0]["y"],
        ]
    )


def point_2d_to_latent(click, settings):
    projector = Reducer(repr=settings["projector"])
    return projector.reconstruct([click])[0]


def latent_point_string_eval(string):
    if string[0] == "[":
        string = string[1:]
    if string[-1] == "]":
        string = string[:-1]
    point = []
    for f in string.split(", "):
        point.append(float(f))
    return np.array(point)


def click_data_to_point_latent(click_data):
    return latent_point_string_eval(click_data["points"][0]["text"])


### Components
def global_view_points(manifold, options):
    train_loader, test_loader = manifold.get_datasets(options["batch-size"])
    data = Data()
    print(data.data)
    for i, (observation, label) in enumerate(train_loader):
        if i * options["batch-size"] >= options["limit"]:
            break
        latent, log_var = manifold.encode(observation)
        measure = manifold.measure(latent)
        recon = manifold.decode(latent)
        loss = manifold.loss_function(recon, observation, latent, log_var)
        data.add_set(
            latent=latent,
            measure=measure,
            observation=observation,
            label=label,
            loss=loss,
        )
    data.add_neigbours()
    return data.get_data()


def global_dimensionality_reducion(data, options):
    points = np.concatenate([t["values"]["latent"] for t in data["sets"]], axis=0)
    projector = Reducer(points)
    options["projector"] = projector.to_string()
    return options


def point_info(click_data, manifold):
    return go.Figure(manifold.point_info(click_data_to_point_latent(click_data)))


def path_info(component, manifold):
    figs = []
    for s in component["sets"]:
        fig = make_subplots(
            rows=1,
            cols=len(s["values"]["latent"]),
            horizontal_spacing=0.01,
            vertical_spacing=0.01,
        )
        fig.update_layout(height=100)
        fig.update_layout(
            margin=dict(
                l=1,  # left
                r=1,  # right
                t=1,  # top
                b=1,  # bottom
            )
        )
        fig.update_xaxes({"visible": False, "showticklabels": True})
        fig.update_yaxes({"visible": False, "showticklabels": True})
        for j, p in enumerate(s["values"]["latent"]):
            fig.add_trace(manifold.point_info(p), 1, j + 1)
        figs.append(s["info"]["description"])
        figs.append(fig)
    return figs


def add_shortest_path(manifold, data, settings, options):
    beg_latent, beg_2d = select_point_from_input_field(settings, options, prefix="beg-")
    end_latent, end_2d = select_point_from_input_field(settings, options, prefix="end-")
    latent, stats = solve_shortest_path(
        manifold,
        beg_latent,
        end_latent,
        options["levels"],
        options["max_nodes"],
        float(options["tol"]),
        options["T"],
    )
    euclidean_latent = euclidean_shortest_path(
        beg_latent,
        end_latent,
        options["levels"],
    )

    data = Data(data)
    data.add_set(
        latent,
        manifold=manifold,
        mode="lines+markers",
        line_color="darkgreen",
        options=options,
        stats=stats,
        description=f"Approximated shortest path using a Riemannian metric, green, beginning: {beg_latent} -> end: {end_latent}",
    )
    data.add_set(
        euclidean_latent,
        manifold=manifold,
        mode="lines+markers",
        line_color="red",
        description=f"Approximated shortest path using a Euclidean metric, red, beginning: {beg_latent} -> end: {end_latent}",
    )
    return data.get_data(), beg_2d, beg_latent, end_2d, end_latent


def select_point_from_input_field(settings, options, prefix):
    projector = Reducer(repr=settings["projector"])
    if options[prefix + "source"] == "latent value":
        latent = latent_point_string_eval(options[prefix + "latent"])
        point_2d = projector.reduce([latent])[0]
    elif (
        options[prefix + "source"]
        == "x and y in 2d-reduced dimension (dimesinality reduction reconstruction)"
    ):
        point_2d = np.array([options[prefix + "x"], options[prefix + "y"]])
        latent = projector.reconstruct([point_2d])[0]
    else:
        raise Exception
    return latent, point_2d


def select_point_from_view(global_click, local_click, options):
    if options["source"] == "global view":
        click = global_click
    elif options["source"] == "local view":
        click = local_click
    else:
        raise Exception
    return click_data_to_point_latent(click), click_data_to_point_2d(click)


def select_point(
    settings,
    global_click,
    local_click,
    options,
):
    if options["source"] in ["latent value", "reconstruct from xy"]:
        selection = select_point_from_input_field(settings, options, prefix="")
    elif options["source"] in ["global view", "local view"]:
        selection = select_point_from_view(global_click, local_click, options)
    else:
        raise Exception
    return selection


def local_dimensionality_reduction(manifold, selected, case, data):
    neighbours = Neighbours(repr=case["neighbours"])
    latent = np.concatenate([t["values"]["latent"] for t in case["sets"]], axis=0)
    label = np.concatenate([t["values"]["label"] for t in case["sets"]], axis=0)
    idxs = neighbours.neighbours_idxs(
        points=np.expand_dims(selected, axis=0),
        radius=data["radius"],
    )
    local = latent[idxs[0]]
    label = label[idxs[0]]
    reducer = Reducer(points=local)
    quality = reducer.reconstruction_quality(local).mean()
    data["projector"] = reducer.to_string()
    # TODO better quality with exps
    data["quality"] = quality
    data = Data(data)
    data.add_set(local, manifold=manifold, mode="markers", label=label)
    return data.get_data(), quality


def contour_maps_points(manifold, settings, selected, options):
    projector = Reducer(repr=settings["projector"])
    point_view = projector.reduce([selected])[0]
    vector = np.array([options["vector-length"], 0])
    direction_goals = np.array(all_directions(options["angle"], vector)) + point_view
    reconstructed = projector.reconstruct([*direction_goals, point_view])
    directions, point_latent_recontructed = reconstructed[:-1], reconstructed[-1]
    directions = directions - point_latent_recontructed

    solver_args = [
        "method",
        "radius",
        "first_step",
        "max_step",
        "rtol",
        "atol",
        "lband",
        "uband",
        "min_step",
    ]
    solver_options = {a: options[a] for a in solver_args if options[a] is not None}

    contour, stats = zip(
        *[
            solve_expmap(
                manifold,
                np.array(selected),
                d,
                options["radius"],
                options["levels"],
                solver_options,
            )
            for d in directions
        ]
    )

    data = Data()
    if not options["vertical"] and not options["horizontal"]:
        for c, s in zip(contour, stats):
            data.add_set(
                c,
                manifold=manifold,
                options=options,
                stats=s,
                description=f"Vertical contour, beginning: {selected} -> direction: {d}",
            )

    if options["vertical"]:
        for c, s, d in zip(contour, stats, directions):
            data.add_set(
                c,
                manifold=manifold,
                mode="lines+markers",
                options=options,
                stats=s,
                description=f"Vertical contour, beginning: {selected} -> direction: {d}",
            )

    if options["horizontal"]:
        contour = np.array(contour)
        # reshape and add circle end
        contour = [
            np.concatenate((contour[:, level], contour[:1, level]), axis=0)
            for level in range(options["levels"])
        ]
        for l, c in enumerate(contour):
            data.add_set(
                c,
                manifold=manifold,
                mode="lines+markers",
                options=options if not options["vertical"] else None,
                description=f"Horizontal contour, level: {l}, first two directions: {directions[0]}, {directions[1]}",
            )

    return data.get_data()


def add_expmap_path(manifold, settings, selected, data, options):
    projector = Reducer(repr=settings["projector"])
    point_view = projector.reduce([selected])[0]
    direction = np.array([options["dir-x"], options["dir-y"]])
    direction_goal = point_view + direction
    direction, point_latent_recontructed = projector.reconstruct(
        [direction_goal, point_view]
    )
    direction = direction - point_latent_recontructed

    solver_args = [
        "method",
        "radius",
        "first_step",
        "max_step",
        "rtol",
        "atol",
        "lband",
        "uband",
        "min_step",
    ]
    solver_options = {a: options[a] for a in solver_args if options[a] is not None}
    latent, stats = solve_expmap(
        manifold,
        point_latent_recontructed,
        direction,
        options["radius"],
        options["levels"],
        solver_options,
    )

    euclidean_latent = euclidean_direction(
        point_latent_recontructed,
        direction,
        options["radius"],
        options["levels"],
    )

    data = Data(data)
    data.add_set(
        latent,
        manifold=manifold,
        mode="lines+markers",
        line_color="darkgreen",
        options=options,
        stats=stats,
        description=f"Approximated directed geodesic using a Riemannian metric, green, {selected} -> direction: {direction}",
    )
    data.add_set(
        euclidean_latent,
        manifold=manifold,
        mode="lines+markers",
        line_color="red",
        description=f"Approximated directed geodesic using a Euclidean metric, red, {selected} -> direction: {direction}",
    )
    return data.get_data()
