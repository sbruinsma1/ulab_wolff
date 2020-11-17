# -*- coding: utf-8 -*-
"""
@author: wolffbs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import mainFunctions as mf
import plotFunctions as pf 

def circadianImgs(expt,exptType=None,measure=None,plotDates=True,
                  individualImgs=True,groupImgs=True,saveFigs=True,
                  normPercent=99,applyPlotLims=False,zThresh=None,**kwargs):
    
    #%%
    exptType,measure = mf.getExptTypeAndMeasure(expt,exptType,measure)
    plotParams = mf.getPlotParams(expt,exptType,measure) 
    for key,val in kwargs.items():
        plotParams[key] = val
#        
#    if measure in ['pellet','poke']:
#        print('\n\nNote: "normPercent" should be as low as possible for pellet or poke measures\n\n')       
    
    #%% acquire and format data
    data = mf.readExptData(expt,exptType,measure)
    data = mf.removeDataMice(data,expts=plotParams['removeExpts'],groups=plotParams['removeGroups'],
                             aIDs=plotParams['removeIDs'],display=True)
    data = mf.removeDataDays(data,plotParams['removeDays'])
    if applyPlotLims:
        data = mf.applyPlotLims(data,plotParams['plotLims'])

    # normalize
    data = pf.removeOutliers(data,zThresh=zThresh) # will skip if zThresh is None
    if normPercent is not None:
        data = data / data.quantile(normPercent/100)
        data[data > 1] = 1

    # add averages
    if groupImgs:
        dataMeans = data.groupby(mf.getGroups(data),axis=1).mean()
        dataMeans.columns = ['all %s'%c for c in dataMeans.columns]
        data = pd.concat((data,dataMeans),axis=1) # add group means
    if not individualImgs:
        data.drop(columns=[c for c in data.columns if c[:4] != 'all '],inplace=True)
    
    #%% convert to 2d frame for each animal
    def dtNanFill(data):
        dtRange = pd.date_range(data.index[0][1],data.index[-1][1],freq=pd.Timedelta(minutes=1)).to_series()
        days = mf.daysFromDatetimes(dtRange,mf.getDateZero(data))
        idx = pd.MultiIndex.from_arrays((days,dtRange.values)).set_names(['Day','DateTime'])
        dataFill = data.reindex(idx)
        return dataFill
    
    def unstackEachColumn(df):
        idx = df.index.to_frame()
        idx['Time'] = idx['DateTime'].dt.time
        df.index = pd.MultiIndex.from_frame(idx[['Day','Time']])

        # reindex to 6:00am - 5:59am cycles
        startTime = pd.Timestamp(6,unit='h')
        times = pd.date_range(startTime,startTime + pd.Timedelta(days=1),
                              freq=pd.Timedelta(minutes=1),closed='left').to_series().dt.time
                              
        return {c: df[c].unstack().reindex(times,axis=1) for c in df.columns}
    dataFill = dtNanFill(data)
    dates = mf.getUniqueDates(dataFill)
    imgData = unstackEachColumn(dataFill)

    #%% plotting functions
    
    def showImgData(df,plotDates,dates,plotParams):
        fig,ax = plt.subplots(figsize=(5,4))
        cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0,as_cmap=True)
        sns.heatmap(df,cmap=cmap,cbar=False,vmin=0,vmax=1,ax=ax)
        
        def outlineResetDays(ax,resetDays,resetLen):
            if resetDays in [None,[]] or resetLen == 0:
                return
            xlim = ax.get_xlim()
            for rD in resetDays:
                yVals = [dates.index.get_loc(rD),dates.index.get_loc(rD+resetLen)]
                for y in yVals:
                    ax.hlines(y,*xlim,color='k',linestyle=':',linewidth=0.5)
            ax.set_xlim(*xlim)
        outlineResetDays(ax,plotParams['resetDays'],plotParams['resetLen'])

        ax.set_title('%s for %s'%(plotParams['dataType'],key),fontsize=14)        
        for spine in ax.spines.values(): # outline figure window
            spine.set_visible(True)
            spine.set_color('k')
            spine.set_linewidth(0.5)
       
        ax.set_xticks(np.arange(0,1560,120))
        ax.set_xticklabels(np.arange(0,25,2))
        ax.set_xlabel('Zeitgeber Time',fontsize=12)
        
        if plotDates:
            ax.set_yticklabels(dates.values,rotation=0)
            ax.yaxis.label.set_visible(False)
        else:
            ax.set_ylabel('Day',fontsize=12)

        fig.tight_layout()
        return fig,ax       

    def setPath(expt,measure):
        path = '%s/circadianImgs_%s/'%(mf.getExptPath(expt),measure)
        if not os.path.exists(path):
            os.mkdir(path)
        return path
    path = setPath(expt,measure)
    
    def setSaveFileName(key,path):
        if key[:4] == 'all ': # is group average
            return '%s%s.png'%(path,key)
        else: # is individual animal        
            return '%s%s_%s.png'%(path,*key.split(':')[::-1])        

    figs = []
    for key,df in imgData.items():
        fig,ax = showImgData(df,plotDates,dates,plotParams)
        if saveFigs:
            figs.savefig(setSaveFileName(key,path))
            plt.close(figs)
        else:
            figs.append(fig)
    return figs

