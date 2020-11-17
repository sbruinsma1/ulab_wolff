# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import plotFunctions as pf


def getVWRACol(data):
    return [c for c in data.columns if 'time active' in c.lower()][0]

def getData(normalize=False):
    data = pd.read_csv(os.getcwd() + '/data/BDNF/BDNF totals.csv')
    data['VWRA'] /= 180 # convert minutes to hours
    data.insert(5,'Ratio',data['Mature'] / data['Pro.'])
    data.drop(columns='Dimer',inplace=True)
    if normalize:
        data.drop(columns='VWRA',inplace=True)
    else:
        data.drop(columns='Normalized VWRA')
    data.rename(columns={'Mature': 'mature BDNF', 'Pro.':'pro-BDNF','Ymaze':'Y-Maze (%)',
                         'VWRA':'Time Active (hr)','Normalized VWRA':'Time Active (% baseline)'},inplace=True)
    return data

def plotLevels(normalize=False,figsize=(2.5,4),fontsize=16):
    data = getData(normalize=False)
    
    figs = [None,None]
    for i,c in enumerate(['mature BDNF','pro-BDNF']):
        figs[i],ax = plt.subplots(figsize=figsize)
        print(c+': ',end='')
        d = data[[c,'group']].dropna()
        pf.swarmPlot(d[c].values,d['group'].values,yLabel=c,ax=ax,statistic='u',
                     drawSig=True,dotsize=6,fontsize=fontsize)
        figs[i].tight_layout()
    return figs

def plotCorr(normalize=False,correl='pearson',figsize=(4,4),fontsize=16,groupStats=True): 
    data = getData(normalize=normalize)
    comparisons = [['mature BDNF','Y-Maze (%)'],
                   ['pro-BDNF','Y-Maze (%)'],
                   ['mature BDNF','Open Time (%)'],
                   ['pro-BDNF','Open Time (%)'],
                   ['mature BDNF',getVWRACol(data)],
                   ['pro-BDNF',getVWRACol(data)]]    

    figs = [None]*len(comparisons)
    for i,comp in enumerate(comparisons):
        figs[i],ax = plt.subplots(figsize=figsize)
        pf.groupScatterCorr(data['group'],[data[c] for c in comp],labels=comp,
                            correl=correl,ax=ax,groupStats=groupStats,addOverall=True,fontsize=fontsize)
        figs[i].tight_layout()
    return figs


