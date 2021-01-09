"""
Implementation of Dynamic Clustering with Polynomial Regression

Based on the paper "A Novel Evolving Clustering Algorithm with Polynomial Regression 
for Chaotic Time-Series Prediction", by Harya Widiputra, Russel Pears, Nikola Kasabov

pdf: https://core.ac.uk/download/pdf/56362399.pdf
"""

from scipy.spatial import distance
import pandas as pd
import numpy as np
import random
import math
from functools import reduce
import sys
import plotly.express as px
import plotly.graph_objects as go
from mvgavg import mvgavg
import copy

class Cluster:
    def __init__(self, Ccj=None, PFMj=None, PFM=[], PF=[]):
        self._Ccj = Ccj
        self._PFMj = PFMj
        self._PFM = PFM
        self._PF = PF

    @property
    def Ccj(self): return self._Ccj
    @Ccj.setter
    def Ccj(self, val): self._Ccj = val

    @property
    def PFMj(self): return self._PFMj
    @PFMj.setter
    def PFMj(self, val): self._PFMj = val

    @property
    def PFM(self): return self._PFM
    @PFM.setter
    def PFM(self, val): self._PFM = val

    @property
    def PF(self): return self._PF
    @PF.setter
    def PF(self, val): self._PF = val

    def add_member(self, PFi, PFMi): 
        self._PF.append(PFi)
        self._PFM.append(PFMi)


def bic(n, k, rss):
    return n * math.log(rss) + k * math.log(n)

# Mean Squared Error
def mse(Y, Yp): 
    return sum(map(lambda y, yp: (y - yp)**2, Y, Yp)) / len(Y)

# determine best polynomial fit up to degree max_k for some data
def best_polyfit(X, Y, max_k):
    # calculate best polynomial fit for the data slice
    fit = [np.poly1d(np.polyfit(X, Y, k)) for k in range(1, max_k)]
    
    minimum_distance = sys.maxsize 
    min_dist_index = None

    for i in range(len(fit)):
        rss = mse(Y, fit[i](X))
        d = bic(len(Y), max_k, rss)
        if d < minimum_distance:
            minimum_distance = d
            min_dist_index = i
 
    return np.poly1d(fit[i].coef)

# pad a polynomial with "val" up to "limit" coefficients
def pad_poly(p, val, limit):
    padded = ([val] *  (limit - len(p) - 1)) + list(p.coef)
    return np.poly1d(np.array(padded))

# return an estimate of a superposition of polynomials in P using a regression of the averages
# of coefficient_j for all p in P
def super_poly(P, T, max_k):
    padded = list(map(lambda p: list(pad_poly(p, 0, max_k).coef), P))
    super_coef = [0] * max_k

    for c in range(max_k):
        super_coef[c] = np.mean([
            padded[p][c] for p in range(len(padded))
        ])

    return np.poly1d(np.array(super_coef))

# return cosine distance of 2 np.poly1d polynomial objects
def cos_dist(A, B):
    return distance.cosine(A.coef, B.coef)


def DyCPR(data_stream, n=10, max_k=6, Dthr=0.05, train_size=0.8):
    C = [] # clusters

    X = [] 
    Y = []
    PF = []   # best-fit regression for data chunk i
    PFM = [] # best-fit regression for next movement of data chunk i
    T = np.linspace(1, n, 100) # time axis of any given data slice

    # slice the data into chunks of sequential data of a fixed size
    for i in range(len(data_stream) - n): # range(number of time-sequential partitions of length n)
        X.append(data_stream[i:i+n])
        Y.append(data_stream[i:i+n+1])
        PF.append(best_polyfit([i for i in range(n)], data_stream[i:i+n], max_k))
        PFM.append(best_polyfit([i for i in range(n+1)], data_stream[i:i+n+1], max_k))


    # split data into train and test sets
    test_size = (1 - train_size) * len(X)
    test_set = []
    test_set_m = []
    test_fit = []
    test_fit_m = []
    len_t = 0
    len_Y = len(X)
    while len_t < test_size:
        i = random.randint(0, len_Y-1)
        test_set.append(X.pop(i))
        test_set_m.append(Y.pop(i))
        test_fit.append(PF.pop(i))
        test_fit_m.append(PFM.pop(i))
        len_Y = len(X)
        len_t = len(test_set)

    N = len(X)

    # initialize clusters
    C.append(Cluster(Ccj=PF[0], PFMj=PFM[0], PF=[PF[0]], PFM=[PF[0]]))

    # perform DyCPR on the training set
    for i in range(len(PF)):
        PFi = PF[i]
        PFMi = PFM[i]

        # find the nearest cluster to PFi
        nearest_Cj = reduce(
            lambda nearest, Cj: 
                Cj if cos_dist(PFi, Cj.Ccj) < cos_dist(PFi, nearest.Ccj)
                   else nearest,
            C, C[0]
        )

        dist_Cj = cos_dist(PFi, nearest_Cj.Ccj)

        # if no cluster was sufficiently similar, create a new cluster
        if dist_Cj > 2 * Dthr:
            C.append(Cluster(Ccj=PFi, PFMj=copy.deepcopy(PFMi), PF=[PFi], PFM=[PFMi]))
        # otherwise add PFi, PFMi to the nearest cluster, and update PFMj
        else:
            nearest_Cj.add_member(PFi, PFMi)
            nearest_Cj.PFMj = super_poly(nearest_Cj.PFM, T, max_k)

    # predict on test set, and compare to rolling average model
    actual = []
    predicted = []
    predicted_ma = []

    test_sample_figs = []
    cluster_figs = []
    fig_height=500 
    fig_width=500

    # find nearest centroid and make a prediction for each polynomial
    for i in range(len(test_set_m)):
        PFi = test_fit[i]

        nearest_Cj = reduce(
            lambda nearest, Cj: 
                Cj if cos_dist(PFi, Cj.Ccj) < cos_dist(PFi, nearest.Ccj) 
                   else nearest,
            C, C[0]
        )
        
        # predicted value calculated as y_{t+1} = y_{t} + py_{t+1}}, where y_{t} is
        # the second-to-last value in sample test_set_m[i]
        yt = test_set[i][n-1]
        
        # remove the x^0 coeficient of the polynomial
        pre_coef = list(nearest_Cj.PFMj.coef) 
        adjusted_coef = pre_coef[0 : len(pre_coef)-1] 
        adjusted_PFMj = np.poly1d(adjusted_coef)

        pyt1 = adjusted_PFMj(n)
        yt1 = yt + pyt1

        m_a = mvgavg(test_set[i], n)[0]

        actual.append(test_set_m[i][n])
        predicted.append(yt1)
        predicted_ma.append(m_a)

        if i < 5: 
            X = [f for f in range(1, n+2)]
            Y = test_set_m[i]
            Tf = np.linspace(1, n+1, 100)
            BF = test_fit[i](Tf)
            M_PFM = nearest_Cj.PFMj(T)
            M_Cc = nearest_Cj.Ccj(T) 

            fig = go.Figure() 
            fig.add_trace(go.Scatter(name="Actual", mode='lines', x=X, y=Y))
            # fig.add_trace(go.Scatter(name="Best Fit", mode="lines", x=Tf, y=BF))
            fig.add_trace(go.Scatter(name="PFMj", x=Tf, y=M_PFM, mode="lines"))
            # fig.add_trace(go.Scatter(name="Ccj", x=Tf, y=M_Cc, mode="lines"))
            fig.add_trace(go.Scatter(name="DyCPR", mode='markers', x=[n+1], y=[yt1], marker=dict(
                                                        size=10,
                                                        line=dict( 
                                                            color='DarkSlateGray',
                                                            width=2
                                                        )
                                                    )))
            fig.add_trace(go.Scatter(name="Moving Average", mode='markers', x=[n+1], y=[m_a], marker=dict(
                                                        size=10,
                                                        line=dict(
                                                            color='DarkSlateGray',
                                                            width=2
                                                        )
                                                    )))
            fig.update_xaxes(title="Day of slice")
            fig.update_yaxes(title="New deaths per 100k of state population")
            fig.update_layout(height=500, width=800)
            test_sample_figs.append(fig)

    # compare 
    rmse = math.sqrt(mse(actual, predicted))
    rmse_ma = math.sqrt(mse(actual, predicted_ma))
    
    # print('\nRMSE: ' + str(rmse))
    # print('RMSE_ma: ' + str(rmse_ma))

    # print('_____________________________________________________________')
    # print('| actual'.ljust(20) + '| DyCPR'.ljust(20) + '| moving average'.ljust(20) + '|')
    # print('|___________________|___________________|___________________|')
    # for i in range(len(actual)):
    #     print(('| ' + str(round(actual[i], 1))).ljust(20) + ('| ' + str(round(predicted[i], 1))).ljust(20) + ('| ' + str(round(predicted_ma[i], 1))).ljust(20) + '|')
    # print('|___________________|___________________|___________________|')

    for Cj in C:
        fig = go.Figure()

        for PFi in Cj.PF:
            fig.add_trace(go.Scatter(x=T, y=PFi(T), line={'color': 'black', 'width': 1}, mode='lines'))

        fig.add_trace(go.Scatter(x=T, y=Cj.Ccj(T), line={'color': 'mediumseagreen', 'width': 7}, mode='lines'))
        fig.add_trace(go.Scatter(x=T, y=Cj.PFMj(T), line={'color': 'red', 'width': 7}, mode='lines'))
        fig.update_xaxes(title="Day of slice")
        fig.update_yaxes(title="New deaths per 100k of state population")
        fig.update_layout(height=fig_height, width=fig_width, showlegend=False)
        cluster_figs.append(fig)

    return {
        'cluster_figs': cluster_figs,
        'num_clusters': len(C),
        'num_slices': len(X),
        'Cc': [Cj.Ccj for Cj in C],
        'X': X,
        'PF': PF,
        'results': {
            'actual': actual,
            'predicted': predicted,
            'predicted_ma': predicted_ma,
            'rmse': rmse,
            'rmse_ma': rmse_ma,
            'test_sample_figs': test_sample_figs
        }
    }
    

def test_bench():
    data = pd.read_csv("data/covid-state-daily-cases-by-population-denull.csv")
    # split data into a dict of 50 dfs, grouping by state_id
    dfs = dict(tuple(data.groupby('state_id')))

    results_dict = {
        "State": [],
        "Data Range": [],
        "DyCPR RMSE (10 trial average)": [],
        "Moving Average RMSE (10 trial average)": [],
        "Higher Accuracy": [],
    }

    for state_id in dfs:
        print(state_id)
        rmse = []
        rmse_ma = []
        for i in range(10):
            print('start trial ' + str(i))
            dycpr = DyCPR(list(dfs[state_id]['positive_increase_per_100k']))
            rmse.append(dycpr['results']['rmse']),
            rmse_ma.append(dycpr['results']['rmse_ma']),

        mean_rmse = round(np.mean(rmse), 2)
        mean_rmse_ma = round(np.mean(rmse_ma), 2)

        results_dict["State"].append(list(dfs[state_id]['state_name'])[0])
        results_dict["DyCPR RMSE (10 trial average)"].append(mean_rmse)
        results_dict["Moving Average RMSE (10 trial average)"].append(mean_rmse_ma)
        results_dict["Data Range"].append((
                str(math.floor(min(list(dfs[state_id]['positive_increase_per_100k']))))
                + '-' + str(math.ceil(max(list(dfs[state_id]['positive_increase_per_100k']))))
        ))
        results_dict["Higher Accuracy"].append("DyCPR" if mean_rmse < mean_rmse_ma else "Moving Average")

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv('results.csv', index=False)

# test_bench()