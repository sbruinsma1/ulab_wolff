# -*- coding: utf-8 -*-
"""
@author: Brian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mainFunctions as mf
import plotFunctions as pf

def dailify(data,measure,threshold=0): 
    df = data.dropna(axis=0,thresh=len(data.columns)*0.5) # keep rows that are at most half NaN
    df.index = df.index.get_level_values('Day') # drop datetimes from index to groupby day
    
    def calcMean(df):
        return df.groupby(df.index).mean()
    def calcSum(df):
        return df.groupby(df.index).sum(min_count=1) # min_count necessary, otherwise np.nan becomes 0
    def calcNumAboveThresh(df,threshold=0):
        return (df > threshold).groupby(df.index).sum()
    def calcSpeed(df,threshold=0):
        return df[df > threshold].groupby(df.index).mean()

    if measure == 'actTime':
        df = calcNumAboveThresh(df,threshold)
    elif measure == 'speed': #executes if "if block" conditional not true (moves through 1 by 1)
        df = calcSpeed(df,threshold)
    elif any([m in measure.lower() for m in ['pellet','poke','dist','act']]):
        df = calcSum(df) # cogwall or telemetry activity measures
    elif any([m in measure.lower() for m in ['heart','temp']]):
        df = calcMean(df) # other telemetry measures
    else: #if none of elif statements work
        assert False, 'Unknown measure: %s'%measure

    df = pd.concat((df,mf.getUniqueDates(data)),axis=1,join='inner')
    df.index = pd.MultiIndex.from_frame(df.pop('Date').reset_index())
    return df
    
def getDailyData(expt,exptType=None,measure=None,output='dfWide',asDecrease=False,**kwargs):
    assert output in ['dfWide','dfLong'], 'output must be "dfWide", or "dfLong"'
    exptType,measure = mf.getExptTypeAndMeasure(expt,exptType,measure)

    plotParams = mf.getPlotParams(expt,exptType,measure) 
    for key,val in kwargs.items():
        plotParams[key] = val
    
    # get data and params
    data = mf.readExptData(expt,exptType,measure)  
    data = dailify(data,measure,plotParams['actTimeThresh'])
    
    # clean and organize data
    data = mf.removeDataMice(data,expts=plotParams['removeExpts'],groups=plotParams['removeGroups'],
                             aIDs=plotParams['removeIDs'],display=True)
    data = mf.removeDataDays(data,plotParams['removeDays'])

    # normalize data
    data = pf.normalizeData(data,plotParams['normalize'],asDecrease=asDecrease)
    data = mf.applyPlotLims(data,plotParams['plotLims'])

    if output == 'dfLong':
        df = data.reset_index().melt(id_vars=['Day','DateTime'],var_name='column',value_name=plotParams['dataType'])
        df.insert(0,'aID',df['column'].apply(lambda x: x.split(':')[0]))
        df.insert(1,'group',df['column'].apply(lambda x: x.split(':')[1]))
        df.drop(columns='column',inplace=True)
        return df
    return data

#%%############################################################################################
# daily plots
#  - days variable should be integers (days), a list of datetime.dates, or pd.DatetimeIndex
#  - unteseted with days variable as strings, but could work
def plotDailyData(data,plotParams,plotDates=False,figsize=[6.4,4.8],xLabel=None,yLabel=None,
                  fontsize=16,colors=None,blackandwhite=False,legendLoc='lower left'): #args to make malleable (not need to rewrite for each fig/exp)
            
    if plotDates: # import stuff for plotting dates
        import matplotlib.dates as mdate
        pd.plotting.register_matplotlib_converters() #import more outer functions (should do at top of file)

    def makeList(r): # make into a list whether or not it is iterable
        try: #to test or if some cases will print error
            rList = list(r)
        except TypeError: #if try block = error -> executed (crashes without)
            rList = [r]
        return rList #what function outputs (depends on goals for ) 
    
    def setResetLog(resetDays,days): # for separating pre- and post-fatigue periods 
        if resetDays is None:
            return [np.ones(len(days),dtype=bool)] #create list of 1s as long as days
        rDays = makeList(resetDays)
        resetLog = [np.array([d < rDays[0] for d in days])]
        for rInd in range(len(rDays)-1): #asks to execute this for all values in set (e.g. participant_ids = dataframe[colname].unique() -> for participant_id in participant_ids: )
            resetLog.append(np.array([d >= rDays[rInd] for d in days]) & \
                            np.array([d < rDays[rInd+1] for d in days]))
        resetLog.append(np.array([d >= rDays[-1] for d in days]))  
        return np.row_stack(resetLog)  
    resetLog = setResetLog(plotParams['resetDays'],mf.getDays(data))

    # calculate statistics -- generates other arrays/columns/values from other spreadsheets
    groups = mf.getGroups(data) #calls mainFunctions.py
    gNames = pf.groupNames(groups)
    gN = data.groupby(groups,axis=1).count()
    gMeans = data.groupby(groups,axis=1).mean()
    if plotParams['errType'] == 'stdErr': #executes if this is true
        gErr = data.groupby(groups,axis=1).sem()
    elif plotParams['errType'] == 'CI': #executes if "if block" conditional not true (moves through 1 by 1)
        gErr = pf.calcConfInterval(data,alpha=0.95)

    # generate plots    
    def plotData(ax,gNames,gMeans,gN,gErr,resetLog,plotDates):
        plotStyles = pf.getGroupPlotStyles(gNames)
        def setGroupLabel(gName,n):
            return '%s: n=%d'%(gName,n)
        x = mf.getDates(gMeans) if plotDates else mf.getDays(gMeans) #other way to write (recommend 1st)
        color = 'Grayscale' if blackandwhite else 'Color'
        for g in gNames:
            gLabel = setGroupLabel(g,gN[g].mode()[0])
            for rInd,rL in enumerate(resetLog):
                ax.errorbar(x[rL],gMeans.loc[rL,g].values,yerr=gErr.loc[rL,g].values, #actual code to create a figure 
                            color=plotStyles.loc[color,g],
                            linestyle=plotStyles.loc['Line',g],
                            marker=plotStyles.loc['Marker',g],
                            elinewidth=1,capsize=1.5,
                            label=gLabel if rInd==0 else None)

    # set figure formatting
    with plt.style.context(('seaborn-darkgrid')): #with plt.style.context(('ggplot')):      
        fig, ax = plt.subplots(figsize=figsize)
    for spine in plt.gca().spines.values(): # outline figure window
        spine.set_color('k')
        spine.set_linewidth(0.5)
    plotData(ax,gNames,gMeans,gN,gErr,resetLog,plotDates) #execute function above once set (consider order of func creation and utilization)
    
    # gray bars around irrad/chemo days
    def markResets(ax,resetDays,resetLen,plotDates,alpha=0.5): 
        if resetDays is None or resetLen == 0:
            return

        ylim = ax.get_ylim()
        if plotDates:
            xUnit = pd.Timedelta(days=1)
            xVals = [pd.to_datetime(mf.getDateZero(data)) + pd.to_timedelta(rD,unit='days') for rD in resetDays]
        else:
            xUnit = 1
            xVals = resetDays
        
        for x in xVals:
            ax.vlines(x - xUnit/2,*ylim,color='k',linestyle=':',linewidth=0.5)
            if resetLen >= 2:
                ax.vlines(x - xUnit/2 + xUnit*resetLen,*ylim,color='k',linestyle=':',linewidth=0.5)
                ax.fill_between([x - xUnit/2,x - xUnit/2 + xUnit*resetLen],[ylim[0]]*2,[ylim[1]]*2,
                                facecolor='gray',alpha=alpha)
        #ax.set_ylim(*ylim)
    markResets(ax,plotParams['resetDays'],plotParams['resetLen'],plotDates)
    
    # pretty it up (mostly using set matplotlib funcs)
    
    ax.legend(loc=legendLoc)
    if yLabel is not None:
        ax.set_ylabel(yLabel,fontsize=fontsize)

    # relabel x ticks
    dInterval = 1 if len(data) <= 12 else 2
    def setXTicks(ax,days,majorInterval=2):
        import matplotlib.ticker as ticker
        ax.xaxis.set_major_locator(ticker.MultipleLocator(majorInterval))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  
        
    def setDateTicks(ax,dInterval):
        ax.tick_params(bottom='on', axis='x', direction='out', length=4)
        fig.autofmt_xdate()
        plt.gca().xaxis.set_major_locator(mdate.DayLocator(interval=dInterval)) 
        def removeOuterLabels(ax):
            plt.tight_layout() #  get_xticks() doesn't seem to work until after plt.tight_layout()
            xTickLabels = [item.get_text() for item in ax.get_xticklabels()]
            if all(x == '' for x in xTickLabels[0]):
                print('warning: xTickLabels were not gotten properly')
                return
            xTickLabels[0],xTickLabels[-1] = '',''
            ax.set_xticklabels(xTickLabels)
        removeOuterLabels(ax)

    if plotDates:
        ax.set_xlim(mf.getDates(data).iloc[0] - pd.Timedelta(days=1),
                    mf.getDates(data).iloc[-1] + pd.Timedelta(days=1))
        setDateTicks(ax,dInterval)  
    else:
        ax.set_xlabel('Day',fontsize=fontsize)
        ax.set_xlim(mf.getDays(data)[0] - 1,mf.getDays(data)[-1] + 1)
        setXTicks(ax,np.arange(len(data)),majorInterval=dInterval)        
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fontsize-4)
    return fig,ax

def getPostResetMean(data,resetDay,resetLen,daysPost):
    if resetDay in [None,[]]: #if no reset, take mean of all data
        return None,data.mean()
    elif isinstance(daysPost,int): # take mean of this many days after reset
        dataPost = data.loc[resetDay+resetLen-1:].iloc[:daysPost].mean()
        dataPre = data.loc[:resetDay].iloc[-daysPost:].mean()
    elif isinstance(daysPost,(list,range)): # take mean of these listed days
        dataPost = data.loc[np.in1d(data.index.get_level_values('Day'),daysPost),:].mean()
        dataPre = data.loc[:resetDay].iloc[-len(daysPost):].mean()
    else:
        assert False, '%s is invalid value for daysPost'%str(daysPost)
    return dataPre,dataPost

def combineData(dataPre,dataPost,plotPre,groupJoin=' '):
    if not plotPre or dataPre is None:
        return dataPost,mf.getGroups(dataPost)
    else:
        dataBoth = pd.concat((dataPre,dataPost))
        groupsBoth = np.array([groupJoin.join([g,s]) for s in ['pre','post'] for g in mf.getGroups(dataPost)])
        return dataBoth,groupsBoth

def standardPlots(expt=None,exptType=None,measure=None, #imp for allFigures.py
                  data=None, # if data is None, will automatically load data for expt  
                  dailyFigsize=[6.4,4.8], plotDates=False, # for daily plot
                  swarmFigsize=[3,4.8],daysPost=6,plotPre=False,dotsize=7, # for boxplot
                  statistic='t',multcomp=False,drawSig=False, # for boxplot statistics
                  fontsize=14,asDecrease=False,blackandwhite=False,**kwargs): # for both plots
    assert expt is not None or data is not None, 'must define either expt or data'
    
    # get default parameters and overwrite with any kwargs (i.e. optional arguments)
    exptType,measure = mf.getExptTypeAndMeasure(expt,exptType,measure)
    plotParams = mf.getPlotParams(expt,exptType,measure) 
    for key,val in kwargs.items():
        plotParams[key] = val
        
    assert not plotPre or not plotParams['normalize'], \
        'cannot set plotPre=True if normalizing (pre data will all be 100%)'
    if data is None:
        data = getDailyData(expt,exptType,measure,asDecrease=asDecrease,**kwargs)        
        data,plotParams['units'] = pf.adjustUnits(data,plotParams['units'],plotParams['normalize'])
        
    yLabel = pf.setYLabel(plotParams['dataType'],units=plotParams['units'],
                          normalize=plotParams['normalize'],asDecrease=asDecrease)

    fig,fig2 = None,None    
    if dailyFigsize is not None:
        fig,ax = plotDailyData(data,plotParams,plotDates,yLabel=yLabel,
                               figsize=dailyFigsize,fontsize=fontsize,blackandwhite=blackandwhite)
        fig.tight_layout()
    
    # plot post-irradiation data as a single statistic
    if swarmFigsize is not None:
        fig2,ax2 = plt.subplots(figsize=swarmFigsize)     
        dataPre,dataPost = getPostResetMean(data,plotParams['resetDays'][0],plotParams['resetLen'],daysPost)
        dataPlot,groupsPlot = combineData(dataPre,dataPost,plotPre)
        pf.swarmPlot(dataPlot,groupsPlot,yLabel=yLabel,ax=ax2,
                     statistic=statistic,multcomp=multcomp,drawSig=drawSig,showN=True,
                     dotsize=dotsize,fontsize=fontsize,blackandwhite=blackandwhite)
        fig2.tight_layout()
    
    return fig,fig2
