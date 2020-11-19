# -*- coding: utf-8 -*-
"""
For Reversal Learning

@author: wolffbs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mainFunctions as mf
import plotFunctions as pf

import matplotlib.dates as mdate
pd.plotting.register_matplotlib_converters()

def getPath():
    return os.getcwd() + '/../data/phenotyper/allCogWall/'
	
def getIrradDates(exptName):
    fileName = getPath() + 'irradDates.txt'
    irradDates = pd.read_csv(fileName,sep=':',index_col=0,header=None).squeeze()
    try:
        return pd.Timestamp(irradDates[exptName]).date()
    except KeyError:
        print('\n\n\tNo irradDates listed for %s!\n\n'%exptName)
        return [None]

def getSegmentFromDay(day):
    segDays = [[-4,-3],[-2,-1],[0,1,2],[3,4],[5,6,7,8,9]]
    for seg,days in enumerate(segDays):
        if day in days:
            return seg+1 # segments start at 1

def getSegName(seg):
    segments = [None,'Baseline','Training','Irradiation','Retraining','Reversal Learning']
    return segments[seg]

def getAllUIDsAndGroups():
    files = [f for f in os.listdir(getPath() + 'pokeLists/') if f[-4:] == '.csv']
    uIDs,expts,arenas,aIDs,groups = zip(*[f.split('.')[0].split('_') for f in files])
    return np.array(uIDs),np.array(groups),np.array(aIDs).astype(int)
	
def getSegPerf(uID,segment,window=30,plot=False):
    readPath = getPath() + 'pokeLists/'
    def getData(uID,readPath):
        fileName = [f for f in os.listdir(readPath) if f[:2] == '%02d'%int(uID)][0]
        origExpt = fileName.split('_')[1]
        return origExpt,pd.read_csv(readPath + fileName,parse_dates=['datetime'])
    origExpt,pokeList = getData(uID,readPath)
    pokeList = pokeList[pokeList['segment'] == segment].reset_index(drop=True)
    pokeList['perf'] = pokeList['correct'].rolling(window).sum() * 100/window
    pokeList['hours'] = (pokeList['datetime'] - pokeList['datetime'][0]) / pd.Timedelta('1h')
    pokeList['expt'] = origExpt

    days = mf.daysFromDatetimes(pokeList['datetime'],resetDate=getIrradDates(origExpt))
    pokeList.index = pd.MultiIndex.from_arrays((days,pokeList['datetime']),names=['Day','DateTime'])
    mf.zeroDates(pokeList) # align dates - this is easier than using day/time combinations
    
    if plot:
        with plt.style.context(('seaborn-darkgrid')):
            fig,ax = plt.subplots(2)
        pokeList.plot(x='hours',y='perf',ax=ax[0])
        pokeList.plot(y='perf',ax=ax[1])
        ax[1].set_xlabel('Poke #')
        for a in ax:
            a.set_ylabel('Performance (%)')   
    return pokeList


################################################################
#%% perf line plots

def getSegGroupPerf(segment=5,window=30,numNanOkay=1):
    # returns a list of dataframes with two elements: first is means, second is error
    # for each item, the first dataframe is perf at each time point, the second is perf at each poke
    assert segment in [2,4,5], 'segment must be 2, 4, or 5'
    uIDs,groups,aIDs = getAllUIDsAndGroups()

    def getPerfDFs(uIDs,segment,timeIndex='hours',window=30):
        perfPoke,perfTime = [],[]
        for uID in uIDs:
            pokeList = getSegPerf(uID,segment,window=window)
            perfPoke.append(pd.Series(pokeList.reset_index()['perf'],name=uID))
            pTindex = pokeList.index.droplevel('Day') #multiindex concat is slooooow
            perfTime.append(pd.Series(index=pTindex,data=pokeList['perf'].values,name=uID))
        perfPoke = pd.concat(perfPoke,axis=1,sort=True).dropna(how='all')
        perfTime = pd.concat(perfTime,axis=1,sort=True).fillna(method='ffill').dropna(how='all')
        
        # return to multiindex
        days = mf.daysFromDatetimes(perfTime.index.to_series(),pd.Timestamp('1950-06-01').date())
        perfTime.index = pd.MultiIndex.from_arrays((days,perfTime.index),names=['Day','DateTime'])

        def removeRedundant(df):
            diffFromPrevRow = (df.shift(-1) != df).any(axis=1)
            diffFromNextRow = (df.shift(1) != df).any(axis=1)
            return df.loc[diffFromPrevRow | diffFromNextRow,:]
        return {'Time': removeRedundant(perfTime), 'Pokes': removeRedundant(perfPoke)}

    # [pokes, time]
    perf = getPerfDFs(uIDs,segment,window=window)
    groupPerf = {g: p.dropna(thresh=p.shape[1]-numNanOkay).groupby(groups,axis=1).mean() for g,p in perf.items()}
    groupErr = {g: p.dropna(thresh=p.shape[1]-numNanOkay).groupby(groups,axis=1).sem() for g,p in perf.items()}
    return groupPerf,groupErr

def plotSegGroupPerf(gMeans,gErr,segment=5,fontsize=14,figsize=(6,4.8)):
    xType = 'Time' if isinstance(gMeans.index,pd.MultiIndex) else 'Pokes'
    if xType == 'Time':
        x = mf.getDates(gMeans)
        fig,ax = plt.subplots(figsize=figsize)
    else:
        x = gMeans.index
        with plt.style.context(('seaborn-darkgrid')):
            fig,ax = plt.subplots(figsize=figsize)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(0.5)     
        
    for g,c in zip(gMeans.columns,[pf.setGroupColor(g) for g in gMeans.columns]):
        ax.plot(x.values,gMeans[g].values,color=c)
        ax.fill_between(x.values,
                        gMeans[g].values - gErr[g].values,
                        gMeans[g].values + gErr[g].values,
                        facecolor=c,alpha=0.3)
    ax.set_ylabel('Performance (%)',fontsize=fontsize)
    if segment is not None:
        ax.set_title(getSegName(segment),fontsize=fontsize+2)
        
    if xType == 'Time':
        ax.set_xlabel('Day',fontsize=fontsize)
        ylim = ax.get_ylim()
        ax.xaxis.set_major_locator(mdate.HourLocator(byhour=6))
        ax.set_xticklabels(mf.getUniqueDates(gMeans,hour=6).index.values)
        
        for d in mf.getUniqueDates(gMeans,hour=18).values: # for each day
            ax.fill_between([pd.to_datetime(d) + t for t in [pd.Timedelta('18h'),pd.Timedelta('30h')]],
                             [ylim[0]]*2,[ylim[1]]*2,facecolor='k',alpha=0.1)
        ax.set_ylim(*ylim)
        ax.set_xlim(x.iloc[0],x.iloc[-1])
    else:
        ax.set_xlabel('Pokes',fontsize=fontsize)
        ax.set_xlim(x[0],x[-1])
            
    return fig,ax
        
def plotSegsGroupPerfPokesTime(segments=[2,5],figsize=(5,4),fontsize=14,window=30):
    figax = []
    for seg in segments:
        groupPerf,groupErr = getSegGroupPerf(seg,window=window)
        for x in groupPerf.keys():
            figax.append(plotSegGroupPerf(groupPerf[x],groupErr[x],seg,
                                        figsize=figsize,fontsize=fontsize))
    for fig,ax in figax:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize-4)
        fig.tight_layout()
    return [f for f,a in figax]
        
#%% get data starting at night
        
def nightData(df,day,endHour=6): # returns data from 18:00 on "day" to endHour on the following day
    dates = mf.getDates(df)
    uDates = mf.getUniqueDates(df)
    ind0 = ((dates.dt.date==uDates[day]) & (dates.dt.hour>=18)).to_numpy().nonzero()[0][0]
    if endHour is None:
        ind1 = len(df) # if no endHour, return the rest of the data
    else:
        ind1 =((dates.dt.date<=uDates[day] + pd.Timedelta('1d')) & 
               (dates.dt.hour<endHour)).to_numpy().nonzero()[0][-1] + 1
    return df.iloc[ind0:ind1,:]

def perfNightData(day,endHour=6,output='perf'):
    assert output in ['perf','len'], 'output must be  "perf" or "len"'
    segment = getSegmentFromDay(day)                  
    uIDs,groups,aIDs = getAllUIDsAndGroups()

    def perfDayDictToSeries(perfDay,output='perf'):
        assert output in ['perf','len'], 'output must be  "perf" or "len"'
        if output == 'perf':
            fun = lambda df: df['correct'].sum()/len(df)
        elif output == 'len':
            fun = lambda df: len(df)
        def getExpt(df):
            return df['expt'].unique()[0]
        perf = {'%s-%s'%(getExpt(val),key): fun(val) for key,val in perfDay.items()}
        return pd.Series(perf,name=output)
    perfDay = {'%s:%s'%(aID,g): nightData(getSegPerf(uID,segment),day,endHour) for uID,g,aID in zip(uIDs,groups,aIDs)}
    perf = perfDayDictToSeries(perfDay,output=output)
    return perf

def allDataNight(day,endHour=6): # starting at 6pm, ending at endHour
    uIDs,groups,aIDs = getAllUIDsAndGroups()
    dist = nightData(mf.readExptData('allCogWall'),day,endHour).sum() / 100 # convert cm to m
    food = nightData(mf.readExptData('allCogWall',measure='pellet'),day,endHour).sum()
    perf = perfNightData(day,endHour,output='perf')

    data = pd.concat((dist,food,perf),axis=1)
    data.columns = ['Distance','Pellets','Perf']
    return data  

####################################################################
#%% swarms and correlations

def plotNumPokes(day=5,endHour=6):
    numPokes = perfNightData(day,endHour,output='len')
    fig,ax = plt.subplots(figsize=(2.5,4))
    pf.swarmPlot(numPokes,mf.getGroups(numPokes),yLabel='Number of Pokes',fontsize=16,ax=ax)
    fig.tight_layout()
    
    groups = mf.getGroups(numPokes)
    print('\n\tAvg number of pokes over first night (until %d:00):'%endHour)
    print('\t\tOverall: %.0f'%numPokes.mean())
    for g in pf.groupNames(groups):
        print('\t\t%s: %.0f'%(g,numPokes.groupby(groups).mean()[g]))
            
def plotPerfNumPokes(day=5,dayPre=-2,numPokes=1536,figsize=(2.5,4),fontsize=16): 
    # mean for Sham group on first night is 1536 pokes (calculated in plotNumPokes())
    perf = perfNightData(5,endHour=None)
    fig,ax = plt.subplots(figsize=(2.5,4))
    pf.swarmPlot(perf,mf.getGroups(perf),yLabel='Performance (%)',fontsize=fontsize,dotsize=7,ax=ax)
    fig.tight_layout()
    return fig

def plotPerfFirstNight(dayPost=5,dayPre=None,plotPre=False,figsize=(2.5,4),fontsize=16):
    # set dayPre to plot the change in performance from dayPre to dayPost
    assert not plotPre or dayPre is not None, 'if plotPre, then dayPre must not be none'

    data = perfNightData(dayPost,endHour=6)            
    if plotPre:
        groups = np.char.add(mf.getGroups(data),' post')
        dataPre = perfNightData(dayPre,endHour=6)
        groupsPre = np.char.add(mf.getGroups(dataPre),' pre')
        fig,ax = plt.subplots(figsize=(4,4))
        pf.swarmPlot(pd.concat((dataPre,data)).values,np.concatenate((groupsPre,groups)),
                     yLabel='Performance (%)',fontsize=fontsize,dotsize=7)
    else:
        data = perfNightData(dayPost,endHour=6)
        groups = mf.getGroups(data)
        yLabel = 'Performance (%)'
        if dayPre is not None:
            data = data - perfNightData(dayPre,endHour=6)
            yLabel = 'Performance change (%)'
        fig,ax = plt.subplots(figsize=figsize)
        pf.swarmPlot(data.values * 100,groups,yLabel=yLabel,ax=ax,
                     fontsize=fontsize,dotsize=7,statistic='t',drawSig=True)        
    fig.tight_layout()
    return fig  

def corrPerf(endHour=6,correl='pearson',figsize=(4,4),fontsize=16,segment=5,day=5):

    data = allDataNight(day,endHour=endHour)
    dataGroups = np.array([i.split(':')[1] for i in data.index])

    fig1,ax = plt.subplots(figsize=figsize)
    pf.groupScatterCorr(dataGroups,data[['Distance','Perf']].values,['Distance (m)','Performance (%)'],
                        correl=correl,fontsize=fontsize,groupStats=True,addOverall=True,ax=ax)
    fig2,ax = plt.subplots(figsize=figsize)
    pf.groupScatterCorr(dataGroups,data[['Pellets','Perf']].values,['Pellets','Performance (%)'],
                        correl=correl,fontsize=fontsize,groupStats=True,addOverall=True,ax=ax)
    fig3,ax = plt.subplots(figsize=figsize)
    pf.groupScatterCorr(dataGroups,data[['Distance','Pellets']].values,['Distance (m)','Pellets'],
                        correl=correl,fontsize=fontsize,groupStats=True,addOverall=True,ax=ax)
    return fig1,fig2,fig3
    