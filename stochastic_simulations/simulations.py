from collections import Callable
import numpy.random as npr
import numpy as np
import plotly.graph_objects as go
from typing import Callable, Tuple, List, Union


def simulation_1(a, b, time, n, x0=0, m=1, plot=False, milstein=False):
    """Simulates random path of stochastic process dx=a*dt+b*dz for the given parameters:
    'a' is drift function and 'b' is diffusion function. They have to be given as user defined or lambda functions with
    two parameters - time and value (they have to be given in that order). If a is a function only of one parameter or
    constant define it as follows:
            - example 1 - a and b are constants: lambda t,x: 0.05;
            - example 2 - a and b are functions only of one parameter: lambda t,x: x**2;
            - example 3 - a and b are functions of both parameters: lambda t,x: t*x;
    'x0' is initial value of the process. It has to be given as an integer or float.
    'time' is list given as tuple [t0,tn] - initial and final moment in time
    'n' is a number of time steps between t0 and t1. It has to be given as an integer.
    'm' is number of processes to be created. It has to be an integer.
    'plot' is optional argument which indicates whether process should be given back as list or plotted on the graph.
    """
    if str(type(a)) != "<class 'function'>" or str(type(b)) != "<class 'function'>":
        print('a and b has to be functions')
    elif type(x0) != int and type(x0) != float:
        return print('Wrongly entered initial value x0')
    elif type(time) != list or len(time) != 2:
        return print("'time' argument has to be list given as a pair [t0,tn]")
    elif type(n) != int or type(m) != int:
        return print("'n' and 'm' have to be integers")
    else:
        dt = (time[1] - time[0]) / n
        process = []
        process.append([[time[0], x0]] * m)
        for i in range(n):
            t = time[0] + i * dt
            x = np.array(process)[i, :, 1]
            drift = np.array(list(map(lambda y: a(t, y), x)))
            diffusion = np.array(list(map(lambda y: b(t, y), x)))
            dw = npr.normal(0, np.sqrt(dt), m)
            if milstein:
                db_dx = np.array(list(map(lambda y: numerical_derivative(b, t, y), x)))
                milstein_method_part = 0.5 * np.array(list(map(lambda y: b(t, y), x))) * db_dx * (dw ** 2 - dt)
                new_x = x + drift * dt + diffusion * dw + milstein_method_part
            else:
                new_x = x + drift * dt + diffusion * dw

            process.append((np.array((np.full(m, t + dt), new_x)).T).tolist())
        if plot == False:
            return process
        else:
            fig = go.Figure()
            [fig.add_trace(go.Scatter(x=np.array(process)[:, i, 0], y=np.array(process)[:, i, 1], mode='lines')) for i
             in range(m)]
            fig.update_layout(
                title=dict(font=dict(color='Navy', size=35), text="<b>Simulated values</b>", x=0.5, y=0.9),
                xaxis_title_text="Time", yaxis_title_text='Values', showlegend=False)
            # fig.show()
            return fig.show(), process


def numerical_derivative(func, t, x, h=1e-5):
    """Compute db/dx using central finite difference."""
    return (func(t, x + h) - func(t, x - h)) / (2 * h)


def predictor_corrector_simulation(a, b, time, n, x0=0, m=1, plot=False, milstein=False, predictor=False):
    if str(type(a)) != "<class 'function'>" or str(type(b)) != "<class 'function'>":
        print('a and b has to be functions')
    elif type(x0) != int and type(x0) != float:
        return print('Wrongly entered initial value x0')
    elif type(time) != list or len(time) != 2:
        return print("'time' argument has to be list given as a pair [t0,tn]")
    elif type(n) != int or type(m) != int:
        return print("'n' and 'm' have to be integers")
    else:
        dt = (time[1] - time[0]) / n
        process = []
        process.append([[time[0], x0]] * m)
        for i in range(n):
            t = time[0] + i * dt
            x = np.array(process)[i, :, 1]
            drift = np.array(list(map(lambda y: a(t, y), x)))
            diffusion = np.array(list(map(lambda y: b(t, y), x)))
            dw = npr.normal(0, np.sqrt(dt), m)
            if milstein:
                db_dx = np.array(list(map(lambda y: numerical_derivative(b, t, y), x)))
                milstein_method_part = 0.5 * np.array(list(map(lambda y: b(t, y), x))) * db_dx * (dw ** 2 - dt)
            elif predictor:
                predicted_x = x + drift * dt + diffusion * dw
                next_t = time[0] + (i + 1) * dt
                next_x = x + 0.5 * (drift + np.array(list(map(lambda y: a(next_t, y), predicted_x)))) * dt + \
                         0.5 * (diffusion + np.array(list(map(lambda y: b(next_t, y), predicted_x)))) * dw

            new_x = x + drift * dt + diffusion * dw + (
                milstein_method_part if milstein else 0) if not predictor else next_x

            process.append((np.array((np.full(m, t + dt), new_x)).T).tolist())
        if plot == False:
            return process
        else:
            fig = go.Figure()
            [fig.add_trace(go.Scatter(x=np.array(process)[:, i, 0], y=np.array(process)[:, i, 1], mode='lines')) for i
             in range(m)]
            fig.update_layout(
                title=dict(font=dict(color='Navy', size=35), text="<b>Simulated values</b>", x=0.5, y=0.9),
                xaxis_title_text="Time", yaxis_title_text='Values', showlegend=False)
            # fig.show()
            return fig.show(), process


def parse_function(func_str: str) -> Callable:
    """
    Safely parse user function string into callable.
    Allows 't', 'x', and numpy functions.
    """
    # Create safe namespace with numpy functions
    safe_dict = {
        'np': np,
        '__builtins__': {},
        'exp': np.exp,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'log': np.log,
        'abs': np.abs,
        'max': max,
        'min': min,
    }

    # Create lambda function
    try:
        func = eval(f"lambda t, x: {func_str}", safe_dict)
        return func
    except Exception as e:
        raise ValueError(f"Invalid function syntax: {str(e)}")


## ----------------------------------


def simulation(
        a: Callable[[float, float], float],
        b: Callable[[float, float], float],
        time: Tuple[float, float],
        n: int,
        x0: float = 0,
        m: int = 1,
        plot: bool = False,
        milstein: bool = False,
        predictor_corrector: bool = False
) -> Union[List, Tuple]:
    """
    Simulates random paths of stochastic process: dx = a(t,x)dt + b(t,x)dz

    Parameters:
    -----------
    a : Callable
        Drift function f(t, x)
    b : Callable
        Diffusion function f(t, x)
    time : Tuple[float, float]
        [t0, tn] - initial and final time
    n : int
        Number of time steps
    x0 : float, default=0
        Initial value of process
    m : int, default=1
        Number of paths to simulate
    plot : bool, default=False
        Whether to plot or return data
    milstein : bool, default=False
        Use Milstein scheme (higher order accuracy)
    predictor_corrector : bool, default=False
        Use Predictor-Corrector (Runge-Kutta) scheme (higher order accuracy)
        Note: milstein and predictor_corrector are mutually exclusive

    Returns:
    --------
    List or Tuple
        Process data (and figure if plot=True)
    """
    # Input validation
    if not callable(a) or not callable(b):
        raise TypeError("'a' and 'b' must be callable functions")
    if not isinstance(x0, (int, float)):
        raise TypeError("'x0' must be int or float")
    if not isinstance(time, (list, tuple)) or len(time) != 2:
        raise ValueError("'time' must be [t0, tn]")
    if not isinstance(n, int) or not isinstance(m, int) or n <= 0 or m <= 0:
        raise ValueError("'n' and 'm' must be positive integers")
    if milstein and predictor_corrector:
        raise ValueError("'milstein' and 'predictor_corrector' cannot both be True")

    t0, tn = time
    dt = (tn - t0) / n

    # initialize: shape (n+1, m, 2) where last dim is [time, value]
    process = np.zeros((n + 1, m, 2))
    process[0, :, 0] = t0  # for every path m_i first time  is 0
    process[0, :, 1] = x0  # for every path m_i first value is x0

    # wiener process from random normal distribution
    # dw = np.random.normal(0, np.sqrt(dt), (n, m))

    for i in range(n):
        t = t0 + i * dt
        x = process[i, :, 1]
        drift = np.array([a(t, xi) for xi in x])
        diffusion = np.array([b(t, xi) for xi in x])
        dw = npr.normal(0, np.sqrt(dt), m)

        if not milstein and not predictor_corrector:
            # Euler method
            new_x = x + drift * dt + diffusion * dw
        elif milstein:
            b_prime = np.array([numerical_derivative(b, t, xi) for xi in x])
            new_x = x + drift * dt + diffusion * dw + 0.5 * diffusion * b_prime * (dw**2 - dt)
        elif predictor_corrector:
            # predictor step (we are using standard drift and diffusion)
            predicted_x = x + drift * dt + diffusion * dw

            # corrector step
            next_t = t + dt
            corr_drift = np.array([a(next_t, xi) for xi in predicted_x])
            corr_diffusion = np.array([b(next_t, xi) for xi in predicted_x])

            # avg
            avg_drift = 0.5 * (corr_drift + drift)
            avg_diffusion = 0.5 * (corr_diffusion + diffusion)

            new_x = x + avg_drift * dt + avg_diffusion * dw

        # Store results
        process[i + 1, :, 0] = t + dt
        process[i + 1, :, 1] = new_x

    if not plot:
        return process.tolist()

    # Plot using Plotly
    fig = go.Figure()
    times = process[:, 0, 0]

    for path_idx in range(m):
        values = process[:, path_idx, 1]
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines',
            name=f'Path {path_idx + 1}' if m > 1 else None,
            showlegend=(m > 1)
        ))

    fig.update_layout(
        title="<b>Simulated Stochastic Paths - Euler</b>" if not milstein and not predictor_corrector else "<b>Simulated Stochastic Paths - Milstein</b>" if milstein else "<b>Simulated Stochastic Paths - Predictor-Corrector</b>",
        xaxis_title="Time",
        yaxis_title="Values",
        hovermode='x unified',
        template='plotly_white'
    )

    # fig.show()
    return process.tolist(), fig
