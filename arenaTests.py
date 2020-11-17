# -*- coding: utf-8 -*-
"""
For YMaze and Open Field

@author: wolffbs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import mainFunctions as mf
import plotFunctions as pf

#########################################################################
#%% general functions

def getFileName(expt,exptType):
    path = '%s%s/%s/'%(mf.getMainPath(),exptType,expt)
    return ['%s%s'%(path,f) for f in os.listdir(path) if '1s' in f][0]

def getIDCol(expt):
    return 'ID' if expt != '20170604' else 'Animal'

def getTime(segment_of_test):
    return segment_of_test.apply(lambda x: x.split(' ')[0]).astype(int)

def getGroups(expt,aIDs):            
    gNames = ['Irrad','Sham']
    gDict = {'20170604': [[1,3,5,6,8,9,13],[2,4,7,10,11,12,14]], # ymaze
             '20170905': [[1,4,7,9,12,16],[2,3,5,8,10,11,13,14]], # ymaze
             '20161017': [[20,21,27,29,30,31,33],[22,23,25,26,28,32]], # open field   
             '20170221': [[1,2,3,6,8,10,13,14],[4,5,7,9,11,12,15,16]], # open field   
             '20170403': [[2,4,5,7,8,9,11,13],[1,3,6,10,12,14,15,16]]} # open field
    
    def groupsFromGroupLists(aIDs,gLists,gNames):
        groups = pd.Series(index=aIDs,data=['_None']*len(aIDs))
        aInts = [int(a.split('-')[1]) for a in aIDs]
        for a,aI in zip(aIDs,aInts):
            for gInd,g in enumerate(gNames):
                if aI in gLists[gInd]:
                    groups[a] = g
        return groups    
    return groupsFromGroupLists(aIDs,gDict[expt],gNames)

def removeID(aID,groups,*dfs): 
    for df in dfs:
        df.drop(columns=aID,inplace=True)
    del groups[aID]
    return (groups,*dfs)

def calcPerfDist(perfTime,dist,maxFun=np.nanmax):
    # returns a DataFrame with distance as the index. maxFun determines how high the distance 
    # index goes... nanmax or nanmedian probably make the most sense
    perfDist = pd.DataFrame(data=np.empty(dist.shape),
                            index=np.linspace(0,maxFun(dist.sum()),len(dist)),
                            columns=dist.columns)
    distCum = dist.fillna(0).cumsum()
    for c in dist.columns:
        perfDist[c] = np.interp(perfDist.index,distCum[c],perfTime[c])
    return perfDist

def plotPerfSubplots(means,CIs,ESs,labels,perfName='Performance (%)',fontsize=16,legendFontsize=12):
    def fillLims(m,ci,lims=[0,100]):
        fillMin = m - ci
        fillMax = m + ci
        fillMin[fillMin < lims[0]] = lims[0]
        fillMax[fillMax > lims[1]] = lims[1]
        return fillMin,fillMax
    
    def addEffectSize(ax,es,legend=True,fontsize=16,legendFontsize=12):
        ax2 = ax.twinx()
        es.plot(ax=ax2,color=(0,0.7,0),label='Effect Size')  
        ax2.set_ylim(0,2)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.axes.get_yaxis().set_visible(False)
        if legend:
            ax.legend().set_visible(False)
            ax2.legend(frameon=False,fontsize=legendFontsize,loc='upper left')
            ax2.set_ylabel('Effect Size (d)',fontsize=fontsize)
            ax2.axes.get_yaxis().set_visible(True)

    fig,axes = plt.subplots(1,len(means),sharey=True,figsize=(3*len(means) - 0.5,4))
    colors = {'Irrad': (0.7,0,0), 'Sham': (0,0,0.7)}
    for i,ax,m,ci,es in zip(range(len(axes)),axes,means,CIs,ESs):
        for g in colors.keys():
            ax.fill_between(m.index,*fillLims(m[g],ci[g]),facecolor=colors[g],alpha=0.3)
            ax.plot(m.index.values,m[g].values,color=colors[g],label=g)
        addEffectSize(ax,es,legend=True if i == len(axes)-1 else False,
                      fontsize=fontsize,legendFontsize=legendFontsize)

    for ax,xLabel in zip(axes,labels):
        ax.set_xlabel(xLabel,fontsize=fontsize) 
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ax != axes[-1]:
            ax.legend().remove()
    axes[0].set_ylabel(perfName,fontsize=fontsize)
    axes[0].legend(frameon=False,fontsize=legendFontsize)
    fig.tight_layout()
    return fig

def plotSwarm(ax,data,groups,title,statistic='t',fontsize=14):
    print('\n%s:\n\t'%title,end='')
    pf.swarmPlot(data,groups,ax=ax,yLabel=title,statistic=statistic,fontsize=fontsize,drawSig=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plotSwarms(data,groups,labels,figsize=None,**kwargs):
    if figsize is None:
        figsize=(2.5,4)
    figs = []
    for d,label in zip(data,labels):
        fig,ax = plt.subplots(figsize=figsize)
        plotSwarm(ax,d,groups,label,**kwargs)
        fig.tight_layout()
        figs.append(fig)
    return figs

def calcMean(df,groups): 
    return df.groupby(groups,axis=1).mean()
def calcErr(df,groups):
    return df.groupby(groups,axis=1).sem()
def calcES(df,groups): # effect size 
    return df.groupby(groups,axis=1).mean().diff(axis=1).dropna(axis=1,how='all').abs().squeeze()/df.std(axis=1)

######################################################################################          
#%% Y-Maze stuff

def getYMazeData(removeJumpers=False,distQuantile=0.01,keepAll=True):

    def getYExptData(expt):# ['20170604','20170905']):
        dataRaw = pd.read_csv(getFileName(expt,'YMaze'),delimiter='\t') # read data
    
        def getDistAndArmsData(dataRaw):
            data = dataRaw.loc[:,[c for c in dataRaw.columns if 'time' == c[-4:]]]
            data.columns = [c.replace(': time','').strip() for c in data.columns]
            data['A'] = data[['A','A0']].sum(axis=1)
            data[data['Center']>0] = 0
            data.drop(columns=['A0','Center'],inplace=True)
            
            data.insert(0,'aID',dataRaw[getIDCol(expt)].apply(lambda x: '%s-%02d'%(expt,x)))
            data.insert(1,'Time',getTime(dataRaw['Segment of test']))
            data.insert(2,'Dist',dataRaw['Distance'])
            dist = data[['aID','Time','Dist']].pivot(index='Time',columns='aID',values='Dist')
            
            def checkForMulipleArms(data):
                numLocs = (data[['A','B','C']] > 0).sum(axis=1)
                if (numLocs > 1).any():
                    multipleLocs = data[numLocs > 1,:].iloc[0,:]
                    print('mouse #%d in more than one arm at t = %d'%tuple(multipleLocs[['aID','time']]))
                          
            data['arm'] = data[['A','B','C']].idxmax(axis=1)   
            data.loc[data[['A','B','C']].sum(axis=1) == 0,'arm'] = 'None'
            arms = data[['aID','Time','arm']].pivot(index='Time',columns='aID',values='arm')
            return dist,arms
        dist,arms = getDistAndArmsData(dataRaw)
        if removeJumpers:
            dist,arms = [d.dropna(axis=1) for d in [dist,arms]]
        
        groups = getGroups(expt,arms.columns)
        if keepAll:
            keep = np.ones(len(groups),dtype=bool)
        else:
            keep = (groups != '_None').values
        return dist.loc[:,keep],arms.loc[:,keep],groups.loc[keep]

    expts = ['20170604','20170905']
    dist,arms,groups = list(zip(*[getYExptData(e) for e in expts]))
    dist,arms = [pd.concat(l,axis=1) for l in [dist,arms]]
    groups = pd.concat(groups)
    
    minDist = dist.sum().quantile(distQuantile)
    keep = dist.sum() > minDist
        
    return dist.loc[:,keep],arms.loc[:,keep],groups[keep.values] 
    
def calcYMazePerf(arms,minTime=0,maxEntries=1000):
    # calculates overall performance (one number per animal)
    def getSequences(arms):
        d = arms != arms.shift()
        d[arms.isna()] = False
        seq = {c: arms.loc[d[c],c] for c in arms.columns}
        for c in arms.columns:
            seq[c][len(arms)] = 'None'
            seq[c] = pd.concat((seq[c].reset_index(drop=True),
                                seq[c].reset_index()['Time'].diff().shift(-1)),axis=1)
            seq[c].rename(columns={c:'arm','Time':'duration'},inplace=True)
            seq[c] = seq[c][seq[c]['arm'] != 'None']
            seq[c] = seq[c][seq[c]['duration'] >= minTime]
            seq[c].reset_index(drop=True,inplace=True)
            seq[c] = seq[c].iloc[:maxEntries,:]
        return seq
    seq = getSequences(arms)
    
    def calcPerf(seq):
        perf = {}
        for k,df in seq.items():
            s = df['arm']
            sdf = pd.DataFrame({0:s,1:s.shift(1),2:s.shift(2)}).dropna()
            sdf['alt'] = sdf.apply(lambda x: x.is_unique, axis=1)
            perf[k] = sdf['alt'].sum() / len(sdf)
        return pd.Series(perf)
    perf = calcPerf(seq)
    return seq,perf

def getNumEntries(arms):
    d = arms != arms.shift()
    d[arms.isna()] = False
    d[arms == 'None'] = False
    return d.cumsum() - 1    

def writeYMazeCorrectEntries(minTime=0,maxEntries=1000):
    dist,arms,groups = getYMazeData()
    csvFile = mf.getMainPath() + 'YMaze/correctEntries.csv'
    
    _,yPerf = calcYMazePerf(arms,minTime=0) # method above
    numEntries = getNumEntries(arms)
    
    def getSequences(arms): # copied from above
        d = arms != arms.shift()
        d[arms.isna()] = False
        seq = {c: arms.loc[d[c],c] for c in arms.columns}
        for c in arms.columns:
            seq[c][len(arms)] = 'None'
            seq[c] = pd.concat((seq[c].reset_index(drop=True),
                                seq[c].reset_index()['Time'].diff().shift(-1)),axis=1)
            seq[c].rename(columns={c:'arm','Time':'duration'},inplace=True)
            seq[c] = seq[c][seq[c]['arm'] != 'None']
            seq[c] = seq[c][seq[c]['duration'] >= minTime]
            seq[c].reset_index(drop=True,inplace=True)
            seq[c] = seq[c].iloc[:maxEntries,:]
        return seq
    
    def calcNumCorrect(seq): # copied from calcPerf above
        perf = {}
        for k,df in seq.items():
            s = df['arm']
            sdf = pd.DataFrame({0:s,1:s.shift(1),2:s.shift(2)}).dropna()
            sdf['alt'] = sdf.apply(lambda x: x.is_unique, axis=1)
            perf[k] = sdf['alt'].sum()
        return pd.Series(perf)
    
    def compareArraysIgnoreNans(a1, a2):
        return np.allclose(a1[np.isfinite(a1)],a2[np.isfinite(a2)])
    
    correctEntries = np.zeros(arms.shape,dtype=int)
    for n in range(2,len(arms)): # this loop is crazy slow
        correctEntries[n,:] = calcNumCorrect(getSequences(arms.iloc[:n,:]))
    correctEntriesDF = pd.DataFrame(data=correctEntries,index=arms.index, columns=arms.columns)
    perf = correctEntriesDF/(numEntries-1).replace(-1,np.nan)
    assert compareArraysIgnoreNans(perf.iloc[-1,:].values,yPerf.values), 'perfs do not match'
    correctEntriesDF.to_csv(csvFile)

def getCorrectEntries():
    try:
        return pd.read_csv(mf.getMainPath() + 'YMaze/correctEntries.csv',index_col=0)
    except FileNotFoundError:
        print('run writeYMazeCorrectEntries() to generate data csv file')

def calcPerfTimes(arms):
    return 100 * getCorrectEntries() / getNumEntries(arms)
	
def yMazePlots(minTime=0,maxEntries=1000,figsize=None,fontsize=16,linePlots=True):
    dist,arms,groups = getYMazeData()   
    perfTime = calcPerfTimes(arms) # get performance over time
    
    def dropGroupless(dfs,groups,dropName='_None'):
        badIDs = groups[groups == dropName].index.to_list()
        groups.drop(index=badIDs,inplace=True)
        for df in dfs:
            df.drop(columns=badIDs,inplace=True)
    dropGroupless([dist,arms,perfTime],groups)
    
    groups,dist,arms,perfTime = removeID('20170905-02',groups,dist,arms,perfTime) # remove column of bad data (barely moved)

    # plot distance and performance
    numEntries = getNumEntries(arms).max()
    figs = plotSwarms((dist.sum(),numEntries,perfTime.iloc[-1,:]),groups,
                      ('Y-Maze Distance (m)','Arm Entries','Spontaneous Alternation (%)'),
                      fontsize=fontsize,figsize=figsize)
    
    if not linePlots:
        return
    
    perfDist = calcPerfDist(perfTime,dist,maxFun=np.nanmedian) # get performance over distance
    
    def calcPerfEntries(arms): # get performance over arm entries
        timeEntries = getNumEntries(arms)
        timeCorrect = getCorrectEntries()
        maxNumEntries = timeEntries.values.max()
        entriesCorrect = pd.DataFrame(data = np.nan * np.empty((maxNumEntries,timeEntries.shape[1])),
                                  index = np.arange(timeEntries.values.max()),
                                  columns = timeEntries.columns)
        entryBool = timeEntries.diff().fillna(0).astype(bool)
        for c in timeEntries.columns:
            entriesCorrect[c] = timeCorrect[c].loc[entryBool[c]].reset_index(drop=True)
        return 100 * entriesCorrect.div(entriesCorrect.index,axis=0).drop(index=[0,1])
    perfEntries = calcPerfEntries(arms)
    nEntries = perfEntries.count().groupby(groups).median().min()
    perfEntries = perfEntries.loc[:nEntries,:]
    
    perfMeans = {'Time (s)': calcMean(perfTime,groups),
                 'Distance (m)': calcMean(perfDist,groups),
                 'Arm Entries': calcMean(perfEntries,groups)}
    perfCI = {'Time (s)': calcErr(perfTime,groups),
              'Distance (m)': calcErr(perfDist,groups),
              'Arm Entries': calcErr(perfEntries,groups)} 
    perfES = {'Time (s)': calcES(perfTime,groups),
              'Distance (m)': calcES(perfDist,groups),
              'Arm Entries': calcES(perfEntries,groups)} 
    fig2 = plotPerfSubplots(perfMeans.values(),perfCI.values(),perfES.values(),perfMeans.keys(),
                            perfName = 'Spontaneous Alternation (%)')
    return figs,fig2

#################################################################################################    
#%% Open Field

def getArenaData(exptType,perfType='time'):
    assert exptType in ['OpenField','ZeroMaze'], '''exptType must be in ['OpenField','ZeroMaze']'''
    assert perfType in ['dist','time'], 'perfType must be "dist" or "time"'
    
    def getExpts(exptType):
        allExpts = {'OpenField':['20161017','20170221','20170403'],
                'ZeroMaze':['201710','201804']}
        return allExpts[exptType]
    expts = getExpts(exptType)
    
    def getPerfName(exptType):    
        if exptType == 'OpenField':
            return ['Center Time','Center Distance']
        elif exptType == 'ZeroMaze':
            return ['Open Time','Open Distance']
        
    def getExptData(expt,exptType,perfType):
        dataRaw = pd.read_csv(getFileName(expt,exptType),delimiter='\t') # read data
        
        def getDistanceCol(expt,*args):
            if expt == '20161017':
                return 'Whole arena : distance'
            elif expt in ['20170221','20170403']:
                return 'WholeArena : distance'
            else:
                return 'Distance'
            
        def getPerfCols(expt,perfType):
            perfColDict = {'20161017': ['Center : time','Center : distance'], # open field
                           '20170221': ['Center30cm : time','Center30cm : distance'], # open field
                           '20170403': ['Center30cm : time','Center30cm : distance'], # open field
                           '201710': [['Open zone 1 : time','Open zone 2 : time'],['Open zone 1 : distance','Open zone 2 : distance']], # zero
                           '201804': [['Open zone 1 : time','Open zone 2 : time'],['Open zone 1 : distance','Open zone 2 : distance']]} # zero
            return perfColDict[expt][0] if perfType == 'time' else perfColDict[expt][1]
        
        def arrangeData(expt,dataRaw,getCol):
            def sumCols(d):
                return d.sum(axis=1) if len(d.shape) > 1 else d
            return pd.DataFrame({'aID': ['%s-%02d'%(expt,a) for a in dataRaw[getIDCol(expt)]],
                                 'Time': getTime(dataRaw['Segment of test']),
                                 'Values': sumCols(dataRaw[getCol(expt,perfType)])
                                 }).pivot(index='Time',columns='aID',values='Values')
        distance = arrangeData(expt,dataRaw,getDistanceCol)
        perf = arrangeData(expt,dataRaw,getPerfCols)
        
        def downSample(d,timeBinSize=5):
            d['timeBin'] = (d.index.values/timeBinSize).astype(int)*timeBinSize
            return d.groupby('timeBin').sum()
        distance,perf = [downSample(d) for d in [distance,perf]]
        groups = getGroups(expt,distance.columns) 
        
        keep = (groups != '_None').values
        return distance.loc[:,keep],perf.loc[:,keep],groups.loc[keep]
    
    dist,perf,groups = list(zip(*[getExptData(e,exptType,perfType) for e in expts]))
    dist,perf = [pd.concat(l,axis=1) for l in [dist,perf]]
    groups = pd.concat(groups)
    
    return dist,perf,groups
	
def openFieldPlots(test='OpenField',fontsize=16,figsize=(2.5,4),linePlots=True):
    if test == 'OpenField':
        statistic = 't'
        behaviorLabel = 'Center Time (%)'
        distanceLabel = 'Open Field Distance (m)'
    elif test == 'ZeroMaze':
        statistic = 't'
        behaviorLabel = 'Open Time (%)'
        distanceLabel = 'Zero Maze Distance (m)'
    timeLabel = 'Time (s)'
        
    dist,openTime,groups = getArenaData(test)
    perfTime = 100 * openTime.cumsum().div(openTime.index + 5,axis=0)
    
    # plot distance and performance
    figs = plotSwarms((dist.sum(),perfTime.iloc[-1,:]),groups,(distanceLabel,behaviorLabel),
                      fontsize=16,statistic=statistic)
    if not linePlots:
        return
    
    perfDist = calcPerfDist(perfTime,dist,maxFun=np.nanmedian) # get performance over distance
    
    perfMeans = {timeLabel: calcMean(perfTime,groups),
                 distanceLabel: calcMean(perfDist,groups)}
    perfCI = {timeLabel: calcErr(perfTime,groups),
              distanceLabel: calcErr(perfDist,groups)}
    perfES = {timeLabel: calcES(perfTime,groups),
              distanceLabel: calcES(perfDist,groups)}
    fig2 = plotPerfSubplots(perfMeans.values(),perfCI.values(),perfES.values(),perfMeans.keys(),
                            perfName = behaviorLabel)
    return figs,fig2

#%% vwra

def dayStats(df,d,col):
    dfd = df[df['day'] == d]
    fig,ax = plt.subplots()
    plotSwarm(ax,dfd[col],dfd['group'],col)
    plt.close(fig)
    
def vwraStuff():
    import plotDaily
    plotDaily.standardPlots('cogVWRA',normalize=0,figSize=[6,4],swarmFigSize=[2.5,4],dotsize=5,fontsize=12)    
    df = plotDaily.getDailyData('cogVWRA',normalize=0,output='dfLong')

    for n in range(3,6):
        print('day %d: '%n,end='')
        dayStats(df,n,'Time Active')

def vwraCorr(figsize=(4,4),fontsize=16,groupStats=False):
    def exptReplace(x):
        exptDict = {'YMaze1': '20170604', 'YMaze2': '20170905',
                    'OpenFieldMice1': '20161007', 'OpenFieldMice2': '20170221', 'OpenFieldMice3': '20170403'}
        for k,v in exptDict.items():
            x = x.replace(k,v)
        return x
    
    yDist,arms,yGroups = getYMazeData()   
    yDist = yDist.sum()
    yPerf = calcPerfTimes(arms) # get performance over time 
    yPerf = yPerf.iloc[-1,:]
    
    ofDist,ofPerf,ofGroups = getArenaData('OpenField')
    ofDist = ofDist.sum()
    ofPerf = ofPerf.sum()/1800
    
    def corrVWRA(d,groups,label,figsize=(4,4),fontsize=16):
        import plotDaily
        vwra = plotDaily.getDailyData('cogVWRA',normalize=0,output='dfWide')
        vwra = vwra.iloc[-3:,:].sum()
        vwra.index = vwra.index.map(lambda x: exptReplace(x.split(':')[0]))
        
        data = pd.concat((d,vwra,groups),axis=1,sort=True).dropna()
        data.columns = ['perf','vwra','group']
        fig,ax = plt.subplots(figsize=figsize)
        pf.groupScatterCorr(data['group'].values,[data['perf'].values,data['vwra'].values/60],[label,'Time Active (hr)'],
                            ax=ax,addOverall=True,groupStats=groupStats,correl='spearman',fontsize=fontsize)
        return fig
    
    fig1 = corrVWRA(yDist,yGroups,'Y-Maze Distance (m)',figsize=figsize,fontsize=fontsize)
    fig2 = corrVWRA(yPerf,yGroups,'Spontaneous Alternation (%)',figsize=figsize,fontsize=fontsize)
    fig3 = corrVWRA(ofDist,ofGroups,'Open Field Distance (m)',figsize=figsize,fontsize=fontsize)
    fig4 = corrVWRA(ofPerf,ofGroups,'Open Time (%)',figsize=figsize,fontsize=fontsize)
    return fig1,fig2,fig3,fig4
    