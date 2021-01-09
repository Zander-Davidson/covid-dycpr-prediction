# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import sympy as sym
import math
import sys
from functools import reduce
import plotly.graph_objects as go
from DyCPR import DyCPR


# practicing lambda functions
def sum(X): return reduce(lambda acc, xi: acc + xi, X, 0)
def mean(X): return sum(X) / len(X)

# "Sum of Squared Residuals"
def sum_sq_res(f, X, Y): return sum(map(lambda x, y: (y - f(x))**2, X, Y))

# def sum_sq_res_poly(p, X, Y): return sum(map(lambda x,y: (y - f(x))**2, X,Y))

# "Variation around the mean", or simply, "Variation"
# can be thought of as the average sum of squares per data
# we want to minimize this
def variance(f, X, Y): return sum_sq_res(f, X, Y) / len(X)
 
# "how much of the variation in y is explained by taking x into account"
# note: we could have used variance instead of sum of squares, since the n will divide out.
# If r^2 = 0 (no variance in y is explained by x) iff ss_mean = ss_fit
# "we saw an (r_squared * 100) percent reduction in varation in y once we took x into account"
def r_squared(f, X, Y):
      ss_mean = sum_sq_res(lambda x: mean(X), X, Y)
      ss_fit = sum_sq_res(f, X, Y)
      return 1 - (ss_fit-ss_mean) / (ss_mean)

# F: "the variation explained by the extra params in the fit / the variation not explained by the extra params in the fit"
# p_fit is the number of parameters in the fit line, e.g. y=a0+(a1*x) has 2 params: a0 and a1.
# p_mean is the number of parameters in the mean line (1)
# NOTE: if the fit is good, then F is large
def free_deg(f, X, Y, p_fit, p_mean=1):
      ss_mean = sum_sq_res(f, X, Y)
      ss_fit = sum_sq_res(f, X, Y)
      return ((ss_mean - ss_fit) / (p_fit - p_mean)) / (ss_fit / (len(X) - p_fit))

def poly1d_pretty_print(p):
    x = sym.symbols('x')
    rounded = [round(c, 1) for c in p.coef]
    return sym.printing.latex(sym.Poly(reversed(rounded), x).as_expr())

# A = [a0, a1, ...]
def polynomial(A): return lambda x: sum([A[i] * x**i for i in range(len(A))])

def poly_deriv(A): return lambda x: sum([i * A[i] * x**(i-1) for i in range(1, len(A))])

def derivative(f, a, h=0.01):
      return (f(a+h) - f(a-h)) / (2*h)

def generate_fig_grid(figs, num_rows, num_cols, id_pfx):
    grid = []

    for i in range(1, num_rows+1):
        span_row = []

        for j in range(1, num_cols+1):
            if (i-1)*num_cols + j -1 < len(figs):
                span_row.append(html.Span(
                    id=(id_pfx + ',' + str(i) + ',' + str(j) + '-span'),
                    children=[dcc.Graph(
                        id=(id_pfx + ',' + str(i) + ',' + str(j)),
                        figure=figs[(i-1)*num_cols + j-1],
                        config={'displayModeBar': False}
                    )]
                ))

        grid.append(html.Div(
            id=id_pfx + ',' + str(i),
            children=span_row,
            style={'display': 'flex', 'justify-content': 'center'}
        ))

    return html.Div(
        id=id_pfx + '-container',
        style={'width': '1300px', 'display': 'flex', 'flex-direction': 'column'},
        children=grid
    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



data = pd.read_csv("data/covid-state-daily-cases-by-population-denull.csv")
# split data into a dict of 50 dfs, grouping by state_id
dfs = dict(tuple(data.groupby('state_id')))

models = {}
degrees = {}
r_sq = {}
curve = np.linspace(min(dfs['AR']['day']), max(dfs['AR']['day']), 100)

# days = dict

for state_id in dfs:
      X = list(dfs[state_id]['day'])
      Y = list(dfs[state_id]['positive_increase_per_100k'])
      k = 4
      # models[state_id] = np.poly1d(np.polyfit(X, Y, k))
      # degrees[state_id] = k

      n = len(X)
      minimum = sys.maxsize

      # finding the best degree for the polynomial fit
      # this may not be necessary, since 'NE', 'IL' are only states with k=4, all others have k=3
      while True:
            model = np.poly1d(np.polyfit(X, Y, k))
            ssr = sum_sq_res(polynomial(model), X, Y)
            bick = (n * np.log(ssr)) + (k * np.log(n))

            if bick > minimum:
                  models[state_id] = model
                  degrees[state_id] = k
                  r_sq[state_id] = r_squared(polynomial(model), X, Y)
                  break;
            else: 
                  minimum = bick
                  k = k + 1

day = curve
positive_increase_per_100k = [models[state_id](curve) for state_id in models]

state_id = [state_id for state_id in models]

# df = pd.DataFrame({
#       'day': curve, 
#       'positive_increase_per_100k': [models[state_id](curve) for state_id in models], 
#       'state_id': [[state_id]*50 for state_id in models]
# })

d = {'day': curve}
for state_id in dfs:
      d[state_id] = models[state_id](curve)
df = pd.DataFrame(d)

fig = px.line(
    df, x='day', y=df.columns, range_y=[-10, 230],
    labels={
        'day': "Week of 2020",
        'value': "New deaths per 100k of state population",
        'variable': 'state'
    })
fig.update_layout(height=500, width=800)


app.layout = html.Div(
    style={'display': 'flex', 'flex-wrap': 'wrap', 'flex-direction': 'row', 'justify-content': 'center', 'width': '100%'},
    children=[
        html.H1(
            style={'flex': '1 0 100%'},
            children='Hello Dash'
        ),

        html.Div(
            style={'flex': '1 0 100%'},
            children='''
                Dash: A web application framework for Python.
            '''),

        dcc.Graph(
            style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'flex': '1 0 100%'},
            id='all-curves',
            figure=fig
        ),

        dcc.Dropdown(
            style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'flex': '1 0 100%', 'width': '200px'},
            id='state-dropdown',
            options=[{'label': list(dfs[state_id]['state_name'])[0], 'value': state_id} for state_id in dfs],
            value='AR'
        ),

        html.Div(id='state-scatter-div'),
        html.Div(id='dycpr-figs')
    ]
)

def single_scatter_fig(state_id):
    X = dfs[state_id]['date']
    Y = dfs[state_id]['positive_increase_per_100k']
    degree = degrees[state_id]
    model = models[state_id]
    curve = np.linspace(min(dfs[state_id]['day']), max(dfs[state_id]['day']), len(X))

    fig = go.Figure()
    fig.add_trace(go.Scatter(name="observed", mode='lines', x=X, y=Y))
    fig.add_trace(go.Scatter(name="polynomial fit", x=X, y=model(curve)))
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="New cases per 100k of state population")
    fig.update_layout(height=600, width=700, title='Daily Increase in COVID-19 Cases, ' + list(dfs[state_id]['state_name'])[0])#, yaxis=dict( range=[0, 90] ))
    return fig

@app.callback(
    Output(component_id='state-scatter-div', component_property='children'),
    Input(component_id='state-dropdown', component_property='value')
)
def update_state_scatter(state_id):
    return html.Div(children=[
        "State population age 18+: " + str(list(dfs[state_id]['population_18_plus'])[0]), html.Br(),
        "Total cases (as of 12-11-20): " + str(list(dfs[state_id]['total_positive'])[-1]), html.Br(),
        "Cases per 100k (as of 12-11-20): " + str(list(dfs[state_id]['positive_per_100k'])[-1]), html.Br(),
        "Polynomial fit: " + poly1d_pretty_print(models[state_id]),
        dcc.Graph(
            id='state-scatter',
            figure=single_scatter_fig(state_id)
        )
    ])

@app.callback(
    Output(component_id='dycpr-figs', component_property='children'),
    Input(component_id='state-dropdown', component_property='value')
)
def update_dycpr_grid(state_id):
    dycpr = DyCPR(list(dfs[state_id]['positive_increase_per_100k']))
    return html.Div(children=[
        "RMSE DyCPR: " + str(round(dycpr['results']['rmse'], 1)), html.Br(),
        "RMSE Moving Average: " + str(round(dycpr['results']['rmse_ma'], 1)), html.Br(), html.Br(), html.Br(),
        "Clusters",
        generate_fig_grid(id_pfx='cluster_grid', figs=dycpr['cluster_figs'], num_rows=math.ceil(dycpr['num_clusters'] / 3), num_cols=3),
        "Test samples",
        generate_fig_grid(id_pfx='cluster_grid', figs=dycpr['results']['test_sample_figs'], num_rows=5, num_cols=1)
    ])


if __name__ == '__main__':
    app.run_server(debug=True)