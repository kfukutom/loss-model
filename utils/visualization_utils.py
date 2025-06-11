from utils import *
from ipywidgets import interact
from IPython.display import Markdown
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Derivatives <3
def f(w): return 5 * (w**4) - (w**3) - 5 * (w**2) + 2 * w - 9
def df(w): return 20 * (w**3) - 3 * (w**2) - 10 * w + 2


# create + tangent line
def create_tangent_line(w):
    try:
        slope = df(w)
        intercept = f(w)-slope*w
        return lambda w: intercept + slope + w
    except Exception: raise


# basic 2D function graph
def draw_f():
    ws = np.linspace(-1.25, 1.25, 1000)
    ys = f(ws)

    fig = px.line(x=ws, y=ys)
    fig.update_layout(
        xaxis_title='$w$',
        yaxis_title='$f(w)$',
        title='$f(w) = 5w^4 - w^3 - 5w^2 + 2w - 9$',
        width=800,
        height=500
    )
    return fig

# tangent line + function overlay
def show_tangent(w0):
    fig = draw_f()
    tan_fn = create_tangent_line(w0)

    fig2 = go.Figure(fig.data)
    fig2.add_trace(go.Scatter(
        x=[w0], y=[f(w0)],
        marker={'color': 'red', 'size': 20},
        showlegend=False
    ))
    fig2.add_trace(go.Scatter(
        x=[-5, 5], y=[tan_fn(-5), tan_fn(5)],
        line={'color': 'red'},
        name='Tangent Line'
    ))
    fig2.update_xaxes(range=[-1.25, 1.25])
    fig2.update_yaxes(range=[-12, -4])
    fig2.update_layout(
        title=f'Tangent line to f(w) at w = {round(w0, 2)}<br>Slope of tangent line: {round(df(w0), 5)}',
        xaxis_title=r'$w$',
        yaxis_title=r'$f(w)$',
        showlegend=False,
        width=800,
        height=500
    )
    return fig2

# convexity illustration
def convexity_visual(a, b, t):
    ws = np.linspace(-20, 20, 1000)
    f = lambda x: x**3 - 3*x**2 + 4*x - 1

    fig = px.line(x=ws, y=f(ws)).update_traces(line=dict(width=8))
    fig.update_layout(
        xaxis_title='$w$',
        yaxis_title='$f(w)$',
        width=800,
        height=600
    )

    fig.add_trace(go.Scatter(x=[a, b], y=[f(a), f(b)]))
    fig.add_trace(go.Scatter(x=[(1 - t) * a + t * b], y=[f((1 - t) * a + t * b)], mode='markers'))
    fig.add_trace(go.Scatter(x=[(1 - t) * a + t * b], y=[(1 - t) * f(a) + t * f(b)], mode='markers'))
    fig.update_traces(marker=dict(size=25))
    fig.update_layout(showlegend=False, title=f't = {t}')
    return fig

# 3D surface plot
def make_3D_surface():
    def f(x1, x2):
        return 3 * np.sin(2*x1) * np.cos(2*x2) + x1**2 + x2**2

    lim = 2
    x1_range = np.linspace(-lim, lim, 100)
    x2_range = np.linspace(-lim, lim, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_values = f(x1_grid, x2_grid)

    fig = go.Figure(data=[
        go.Surface(
            x=x1_grid,
            y=x2_grid,
            z=z_values,
            colorscale='RdBu_r',
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True)))
        )
    ])

    fig.add_trace(go.Scatter3d(x=[-2, 2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='gray', width=2)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-2, 2], z=[0, 0], mode='lines', line=dict(color='gray', width=2)))

    fig.update_layout(
        title='$$f(w_1, w_2) = 3  \\sin(2 w_1) \\cos(2 w_2) + w_1^2 + w_2^2$$',
        scene=dict(
            xaxis_title='w1',
            yaxis_title='w2',
            zaxis_title='f(w1, w2)',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        width=800,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
    )
    return fig

# 2D gradient descent path
def show_gd_path_contour(w1_start=-0.5, w2_start=1, step_size=0.1, iterations=10):
    def f(x1, x2):
        return 3 * np.sin(2*x1) * np.cos(2*x2) + x1**2 + x2**2

    def dfx1(x1, x2):
        return 6 * np.cos(2 * x1) * np.cos(2 * x2) + 2 * x1

    def dfx2(x1, x2):
        return -6 * np.sin(2 * x1) * np.sin(2 * x2) + 2 * x2

    lim = 2
    x1_range = np.linspace(-lim, lim, 100)
    x2_range = np.linspace(-lim, lim, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_values = f(x1_grid, x2_grid)

    fig = go.Figure(data=[
        go.Contour(
            z=z_values,
            x=x1_range,
            y=x2_range,
            colorscale='RdBu_r',
            contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))
        )
    ])

    w1, w2 = w1_start, w2_start
    path_x, path_y = [w1], [w2]

    for _ in range(iterations):
        grad_x, grad_y = dfx1(w1, w2), dfx2(w1, w2)
        w1, w2 = w1 - step_size * grad_x, w2 - step_size * grad_y
        path_x.append(w1)
        path_y.append(w2)

    fig.add_trace(go.Scatter(
        x=path_x, y=path_y,
        mode='lines+markers',
        line=dict(color='gold', width=3),
        marker=dict(size=8, color='gold')
    ))

    fig.update_layout(
        title=f'<b><span style="color:gold">Gradient Descent Path</span></b> from ({w1_start}, {w2_start})',
        xaxis_title='w1',
        yaxis_title='w2',
        width=800,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    return fig


# 3D gradient descent path
def show_gd_path_surface(w1_start=-0.5, w2_start=1, step_size=0.1, iterations=10):
    def f(x1, x2):
        return (3 * np.sin(2*x1) * np.cos(2*x2) + x1**2 + x2**2
)
    def dfx1(x1, x2):
        return (6 * np.cos(2 * x1) * np.cos(2 * x2) + 2 * x1)

    def dfx2(x1, x2):
        return( -6 * np.sin(2 * x1) * np.sin(2 * x2) + 2 * x2)

    lim = 2
    x1_range = np.linspace(-lim, lim, 100)
    x2_range = np.linspace(-lim, lim, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_values = f(x1_grid, x2_grid)

    fig = go.Figure(data=[
        go.Surface(
            x=x1_grid,
            y=x2_grid,
            z=z_values,
            colorscale='RdBu_r',
            contours=dict(z=dict(show=True, usecolormap=True, project=dict(z=True)))
        )
    ])

    w1, w2 = w1_start, w2_start
    path_x, path_y = [w1], [w2]
    path_z = [f(w1, w2)]

    for _ in range(iterations):
        grad_x, grad_y = dfx1(w1, w2), dfx2(w1, w2)
        w1, w2 = w1 - step_size * grad_x, w2 - step_size * grad_y
        path_x.append(w1)
        path_y.append(w2)
        path_z.append(f(w1, w2))

    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        line=dict(color='gold', width=2),
        marker=dict(size=8, color='gold')
    ))

    fig.update_layout(
        title=f'<b>Gradient Descent Path</b> from ({w1_start}, {w2_start})',
        scene=dict(
            xaxis_title='w1',
            yaxis_title='w2',
            zaxis_title='f(w1, w2)',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        width=800,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    return fig



# FINAL VIEW:
def display_paths(w1_start=-0.5, w2_start=1, step_size=0.1, iterations=10):
    fig1 = show_gd_path_contour(w1_start, w2_start, step_size, iterations)
    fig2 = show_gd_path_surface(w1_start, w2_start, step_size, iterations)

    traces1 = fig1.data
    #print(traces1)
    traces2 = fig2.data

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "contour"}, {"type": "surface"}]],
        shared_xaxes=True,
        shared_yaxes=True
    )

    for trace in traces1:fig.add_trace(trace, row=1, col=1)
    for trace in traces2: fig.add_trace(trace, row=1, col=2)

    fig.update_layout(
        title_text=fig1.layout.title.text,
        width=1600,
        height=700,
        scene=dict(
            xaxis_title='w1',
            yaxis_title='w2',
            zaxis_title='f(w1, w2)',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        showlegend=False
    )

    assert(fig != None), print(f'Caught an unusual exception.\n')
    try:
        return fig
    except Exception as e: raise(e)
    finally: print(f'Successfully generated.')