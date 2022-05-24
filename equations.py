import time
from typing import Iterable
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from dash import html

# This function evaluates the differential equation c'' = f(c, c')
def geodesic_system(manifold, c, dc):
    # Input: c, dc ( D x N )
    D, N = c.shape
    assert c.shape == dc.shape
    # Metric and the derivative
    M, dM = manifold.metric_tensor(c.T, nargout=2)
    # Output (D x N)
    ddc = np.zeros((D, N))
    # Non-Diagonal Metric Case, M ( N x D x D ), dMdc_d (N x D x D x d=1,...,D)
    for n in range(N):
        Mn = np.squeeze(M[n, :, :])
        if np.linalg.cond(Mn) < 1e-15:
            raise Exception('Ill-condition metric!\n')
        dvecMdcn = dM[n, :, :, :].reshape(D * D, D, order='F')
        blck = np.kron(np.eye(D), dc[:, n])
        # Geodesic equation (7) https://arxiv.org/pdf/1710.11379.pdf
        ddc[:, n] = -0.5 * (
            np.linalg.inv(Mn) @ (
                2 * blck @ dvecMdcn @ dc[:, n] - dvecMdcn.T @ np.kron(dc[:, n], dc[:, n])
            )
        )
        # Alternative Geodesic equation (8) https://arxiv.org/pdf/1710.11379.pdf
        # TODO

    return ddc

# This function changes the 2nd order ODE to two 1st order ODEs takes c, dc and returns dc, ddc.
def second2first_order(manifold, state):
    # Input: state [c; dc] (2D x N), y=[dc; ddc]: (2D x N)
    D = int(state.shape[0] / 2)
    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)
    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    cmm = geodesic_system(manifold, c, cm)  # D x N
    y = np.concatenate((cm, cmm), axis=0)
    return y

# This function returns the boundary conditions
def boundary_conditions(ya, yb, c0, c1):
    D = len(c0)
    retVal = np.zeros(2 * D)
    retVal[:D] = ya[:D] - c0.flatten()
    retVal[D:] = yb[:D] - c1.flatten()
    return retVal

# This is the default solver that is a build-in python BVP solver.
def solve_shortest_path(
        manifold, 
        c0: np.ndarray, 
        c1: np.ndarray, 
        steps: int, 
        max_nodes: int, 
        tol: float,
        T: int, # def 30
    ):
    D = len(c0)
    # The functions that we need for the bvp solver
    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc)  # D x T, implements c'' = f(c, c')
    bc_fun = lambda ya, yb: boundary_conditions(ya, yb, c0, c1)  # 2D x 0, what returns?
    # Initialize the curve with straight line or with another given curve
    t_init = np.linspace(0, 1, T, dtype=np.float32)  # T x 0
    c_init = np.outer(
        c0.reshape(-1, 1), 
        (1.0 - t_init.reshape(1, T)),
    ) + np.outer(
        c1.reshape(-1, 1), 
        t_init.reshape(1, T),
    )  # D x T
    dc_init = (c1 - c0).reshape(D, 1).repeat(T, axis=1)  # D x T
    c_dc_init = np.concatenate((c_init, dc_init), axis=0)  # 2D x T
    # Solve the geodesic problem
    start = time.time()
    solution = solve_bvp(
        ode_fun, 
        bc_fun, 
        t_init.flatten(), 
        c_dc_init, 
        tol=tol, 
        max_nodes=max_nodes, 
        verbose=2
    )
    elapsed = time.time() - start
    ts = np.linspace(0,1,steps)
    print('Geodesic solver (bvp) succed!')
    path = solution.sol(ts)[:D, :].T
    stats = {
        'status': solution.message,
        'number of iterations': solution.niter,
        'success:': solution.success,
        'elapsed': elapsed,
    }
    return path, stats

def euclidean_shortest_path(
        c0: np.ndarray, 
        c1: np.ndarray, 
        steps: int, 
    ):
    return np.array([np.linspace(c0[i], c1[i], steps) for i in range(len(c0))]).T

# This function solves the initial value problem
# for the implementation of the expmap
def solve_expmap(
        manifold, 
        x: np.ndarray, 
        v: np.ndarray, 
        time_range: float, 
        steps: int,
        options,
    ):
    # Input: v,x (D)
    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc).flatten()  
    # The vector now is in normal coordinates
    # The tangent vector lies in the normal coordinates
    required_length = np.linalg.norm(v)  
    # Rescale the vector to be proper for solving the geodesic.
    v = v / required_length
    M = manifold.metric_tensor(x)[0]
    a = (required_length / np.sqrt(v.reshape(1, -1) @ M @ v.reshape(-1, 1))).flatten()
    # The vector now is on the exponential coordinates
    v = a * v
    init = np.concatenate((x, v), axis=0)
    # Solve the IVP problem
    # First solution of the IVP problem
    start = time.time()
    solution = solve_ivp(
        ode_fun, 
        [0, time_range], 
        init, 
        t_eval=np.linspace(0, time_range, steps), 
        # dense_output=True,
        # vectorized=True,
        **options,
    )      
    elapsed = time.time() - start
    print('Geodesic solver (ivp) succed!')
    path = solution.y[:len(x), :].T
    stats = {
        'status': solution.message,
        'Number of evaluations of the right-hand side': solution.nfev,
        'Number of evaluations of the Jacobian': solution.njev,
        'Number of LU decompositions.': solution.nlu,
        'success:': solution.success,
        'elapsed': elapsed,
    }
    return path, stats

def euclidean_direction(
        x: np.ndarray, 
        v: np.ndarray, 
        time_range: float, 
        steps: int,
    ):
    return euclidean_shortest_path(x, x + v * time_range, steps)

def all_directions(n, vector):
    directions = []
    base = complex(*vector)
    mult = complex(np.cos(2 * np.pi / n), np.sin(2 * np.pi / n)) 
    for i in range(n):
        directions.append(np.array([base.real, base.imag]))
        base *= mult
    return directions

def euclidean_neighbors(
    point, 
    inputs_sets, 
    points_sets, 
    measure_sets, 
    radius,
):
    inputs = []
    neighbors = []
    neighbor_measure = []
    for _is, ps, ms in zip(inputs_sets, points_sets, measure_sets):
        for i, p, m in zip(_is, ps, ms):
            if np.linalg.norm(point - p) < radius:
                inputs.append(i)
                neighbors.append(p)
                neighbor_measure.append(m)
    return inputs, neighbors, neighbor_measure
