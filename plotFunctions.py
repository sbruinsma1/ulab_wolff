# -*- coding: utf-8 -*-
"""

@author: Brian
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import mainFunctions as mf

def groupNames(groups):
	# return sorted group names from groups array 
    gNames = np.array(sorted(list(set(groups))))
    
    possibleControls = ['sham','wt','ctrl']
    def moveToEnd(array,itemsToMove):
        toMove = np.zeros(array.shape).astype(bool)
        for gInd,g in enumerate(gNames):
            if any([c in g.lower() for c in itemsToMove]):
                toMove[gInd] = True
        return np.concatenate((array[~toMove],array[toMove]))   
    gNames = moveToEnd(gNames,['4 gy','8 gy'])
    gNames = moveToEnd(gNames,['post'])
    gNames = moveToEnd(gNames,possibleControls)
    return [str(g) for g in gNames]

###################################################################
#%% stats

def adjustUnits(data,units,normalize=0):
    if normalize:
        return data,units
    elif units == 'm':
        data /= 1000
        units = 'km'
    elif units == 'cm':
        data /= 100
        units = 'm'
    elif units == 'min':
        data /= 60
        units = 'hr'
    elif units == 'cm/min':
        data /= 100
        units = 'm/min'
    return data,units
 
# normalize to a number of baseline days/dates or a list of days/dates
def normalizeData(data,normalize,resetDay=0,asDecrease=False):
    if normalize in [0,[],None]:
        return data
    try: 
        iter(normalize)
    except TypeError: # is not iterable
        baseline = (mf.getDays(data) < resetDay) 
        while sum(baseline) > normalize:
            baseline[np.nonzero(baseline)[0][0]] = False
    else: # is iterable
        baseline = np.in1d(mf.getDays(data),normalize) 
    
    if not asDecrease:
        return 100 * data/data.loc[baseline].mean()
    else:
        return 100 * (1 - data/data.loc[baseline].mean())

def removeOutliers(data,zThresh=3,axis=0,display=True):
	# set data to np.nan if absolute value of its z-score exceeds z-thresh
	# will aso return nan if the standard deviation of data is zero
    if zThresh is None:
        return data
    dataZ = (data - data.mean())/data.std()
    outliers = dataZ.abs() > zThresh
    n = outliers.sum().sum()
    print('\t\tremoved %d of %d data points (%.3f%%)'%(n,outliers.size,100*n/outliers.size))
    dataOut = data.copy()
    dataOut[outliers] = np.nan
    return dataOut

def calcConfInterval(data,alpha=0.95):
    def confidenceWidth(vector,alpha=0.95):
        v = vector[~np.isnan(vector)].squeeze() 
        return np.mean(v) - stats.t.interval(alpha, len(v)-1, loc=np.mean(v), scale=stats.sem(v))[0]
    cwFunction = lambda x: x.apply(confidenceWidth,axis=1,alpha=alpha)
    return data.groupby(mf.getGroups(data),axis=1).agg(cwFunction)


########################################################################
#%% general plotting functions

def getGroupPlotStyles(gNames):
    groupStylesDF = pd.read_csv(mf.getMainPath() + 'groupStyles.csv',index_col=0)
    groupStylesDF.fillna(groupStylesDF.loc['default'],inplace=True)
    grayscaleDict = {'black': .1,
                     'dark': .366,
                     'gray': .5,
                     'light': .633,
                     'white': .9}
    colorDict = {'red': [0.7,0,0],
                 'blue': [0,0,0.7],
                 'green': [0,0.7,0],
                 'gray': [0.5,0.5,0.5],
                 'black': [0,0,0]}
    plotStyles = {}
    for g in gNames:
        try:
            plotStyles[g] = groupStylesDF.loc[g.lower()].copy()
        except KeyError: 
            plotStyles[g] = groupStylesDF.loc['default'].copy()
        plotStyles[g]['Color'] = colorDict[plotStyles[g]['Color']]
        plotStyles[g]['Grayscale'] = grayscaleDict[plotStyles[g]['Grayscale']]
    return pd.DataFrame(plotStyles)

def setGroupColor(g):
    return getGroupPlotStyles([g]).loc['Color'][0]

def setYLabel(dataType,units=None,normalize=False,asDecrease=False):
    if normalize:
        if asDecrease:
            yLabel = dataType + ' Decrease (%)'
        else:                
            yLabel = dataType + ' (% baseline)'
    else:
        yLabel,yUnits = dataType,units
        if yUnits not in [None,'']:
            yLabel = '%s (%s)' %(yLabel,yUnits)
    return yLabel


###################################################################################            
#%% swarm plot

def pairwise_ttests(data,groups,comps='all',statistic='t',multcomp=False,oneSided=False):
    assert statistic in 'tu', 'unknown statistic %s'%statistic
    pStat = 'p'
    
    def removeNans(data,groups):
        nans = np.isnan(data)
        return data[~nans],groups[~nans]
    data,groups = removeNans(data,groups)
    
    def printTResults(name1,name2,statistic,pStat,t,p,n1,n2,cohen):
        name1,name2 = [s.replace('\n','-') for s in [name1,name2]]
        if p >= 0.001:
            print('%s vs %s: d = %.2f, %s = %.2f, %s=%.4f, n=%d+%d'%(name1,name2,cohen,statistic,t,pStat,p,n1,n2))
        else:
            print('%s vs %s: d = %.2f, %s = %.2f, %s=%.4e, n=%d+%d'%(name1,name2,cohen,statistic,t,pStat,p,n1,n2))
        
    gNames = groupNames(groups)
    if comps == 'all':
        from itertools import combinations
        comps = list(combinations(gNames,2))
    tpnd = np.empty((len(comps),5))
    for i,c in enumerate(comps):
        n = [sum(groups==c[0]),sum(groups==c[1])]
        d1 = data[groups==c[0]]
        d2 = data[groups==c[1]]
        cohen = (d2.mean() - d1.mean()) / np.concatenate((d1,d2)).std()
        if statistic == 't':
            t,p = stats.ttest_ind(d1,d2,equal_var=False)
        elif statistic == 'u':
            t,p = stats.mannwhitneyu(d1,d2,alternative='two-sided')
        if oneSided:
            p /= 2
        tpnd[i,:] = abs(t),p,n[0],n[1],cohen

    if multcomp:
        from statsmodels.stats.multitest import multipletests
        reject,tpnd[:,1],a1,a2 = multipletests(tpnd[:,1])
        pStat = 'corrected p'

    for i,c in enumerate(comps): 
        printTResults(c[0],c[1],statistic,pStat,*tpnd[i,:])

    return {c: tpnd[i,1] for i,c in enumerate(comps)}

def swarmPlot(data,groups=None,yLabel='Activity',plotType='box',ax=None,
              statistic='t',multcomp=False,oneSided=False,drawSig=False,showN=True,
              blackandwhite=False,alpha=0.5,dotsize=7,fontsize=14,starsize=14):
    
    if ax is None:
        ax = plt.gca()
    elif ax == 'new':
        fig,ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    dF = pd.DataFrame({'Group': groups, yLabel: data.squeeze()})
    gNames = groupNames(groups)
    plotStyles = getGroupPlotStyles(gNames)
    if blackandwhite:
        colors,markerlinewidth = plotStyles.loc['Grayscale'].values,1
    else:
        colors,markerlinewidth = plotStyles.loc['Color'].values,0

    sns.swarmplot(x='Group',y=yLabel,data=dF,size=dotsize,order=gNames,ax=ax,
                  edgecolor='k',palette=colors,linewidth=markerlinewidth)
    if plotType == 'box':
        sns.boxplot(x='Group',y=yLabel,data=dF,order=gNames,palette=colors,linewidth=1,ax=ax)
    elif plotType == 'bar':        
        sns.barplot(x='Group',y=yLabel,data=dF,errwidth=1,order=gNames,palette=colors,ax=ax,
					edgecolor='k'*len(gNames))
    
    def getPatches(ax,plotType):
        if plotType == 'bar':
            return ax.patches
        elif plotType == 'box':
            return ax.artists

    if alpha != 1:
        for patch in getPatches(ax,plotType):
            patch.set_facecolor((*patch.get_facecolor()[:3],alpha))
    
    if blackandwhite:
        def setHatches(ax,hatches):
            for hatch,patch in zip(hatches,getPatches(ax,plotType)):
                if hatch is not None:
                    patch.set_hatch(hatch)        
        setHatches(ax,plotStyles.loc['Hatch'])

    if fontsize is not None:
        def setAllFontSize(ax,fontsize):
            for item in [ax.title,ax.xaxis.label,ax.yaxis.label]:
                item.set_fontsize(fontsize)
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontsize(fontsize-4)
        setAllFontSize(ax,fontsize)

    # x axis labels
    def wordsOnNewlines(label,minWordCharLen=3): # split multi-word group names onto multiple lines
        words = label.split(' ')
        newLabel = words[0]
        for i,word in enumerate(words[1:]):
            if len(word) >= minWordCharLen and len(words[i-1]) >= minWordCharLen:
                newLabel += '\n%s'%word
            else:
                newLabel += ' %s'%word
        return newLabel
    xLabels = [wordsOnNewlines(g) for g in gNames]
    if showN:
        groupN = [sum([g == gName for g in groups]) for gName in gNames]
        xLabels = [g + '\nn=%d'%n for g,n in zip(xLabels,groupN)]
    ax.set_xticklabels(xLabels)
    ax.xaxis.label.set_visible(False)

    if statistic is not None:
        pVals = pairwise_ttests(data,groups,statistic=statistic,multcomp=multcomp,oneSided=oneSided)
        if statistic == 't':
            pShapiro = [stats.shapiro(data[groups==g])[1] for g in gNames]
            print('    shapiro: p = %s'%', '.join(['%.3f'%p for p in pShapiro]))
            
        if drawSig:
            def plotSig(ax,g1,g2,txt='*',yOffset=0.02):
                yLim = ax.get_ylim()
                yOff = yOffset * (yLim[1] - yLim[0])
                y = yLim[1] + yOff
                x = [gNames.index(g) for g in [g1,g2]]
                ax.plot(x,[y]*2,'k')
                for xVal in x:
                    ax.plot([xVal]*2,[y,y-yOff],'k')
                ax.text(np.mean(x),y,txt,fontsize=starsize,horizontalalignment='center')  
            
            sigDict = {0.05: '*', 0.005: '**', 0.0005: '***'}
            def getSigTxt(p,sigDict):
                for key,val in sorted(sigDict.items()):
                    if p < key:
                        return val
                    
            for [g1,g2],p in pVals.items():
                txt = getSigTxt(p,sigDict)
                if txt is not None:
                    plotSig(ax,g1,g2,txt)

###############################################################################
#%% correlation plots

def keepFinite(xInput,yInput):
    validInds = np.isfinite(xInput) & np.isfinite(yInput)
    return xInput[validInds],yInput[validInds]    
        
def scatterCorr(xInput,yInput,ax=None,color='k',gName=None,axLabels=None,correl='pearson',
                fontsize=14,text=True,line=True,scatter=True):
    if ax is None:
        ax = plt.gca()
    
    # deal with nans
    x,y = keepFinite(xInput,yInput)
    
    # plot scatter and 1-degree regression line, write r & p values on plot    
    if scatter:
        ax.scatter(x,y,color=color)
        
    def setLims(x,scaling):
        xRange = max(x) - min(x)
        if type(scaling) is not list:
            scaling = [scaling]*2
        return [min(x) - scaling[0]*xRange,max(x) + scaling[1]*xRange]
    xlims = setLims(x,0.2)
    ylims = setLims(y,[0.1,0.3])
    
    if line:
        def plotRLine(ax,x,y,xlims,ylims):  
            p = np.polyfit(x,y,1)        
            ax.plot(xlims,np.polyval(p,xlims),':',color=color,linewidth=1)
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
        plotRLine(ax,x,y,xlims,ylims)
    
    if text:
        assert correl.lower()[0] in 'ps', 'correl must be pearson or spearman'
        if correl.lower()[0] == 'p':
            r,p = stats.pearsonr(x,y)
        elif correl.lower()[0] == 's':
            r,p = stats.spearmanr(x,y)
        rTxt = ''.join(['\n' for i in range(len(ax.lines))])
        rTxt = rTxt if gName is None else rTxt + '%s: '%gName
        rTxt = rTxt + 'r = %0.02f, p = %.03f, n = %d'%(r,p,len(x))
        ax.text(0.01,1.02,rTxt,color=color,transform=ax.transAxes,verticalalignment='top',
                fontsize=fontsize-4)
    
    if axLabels is not None:
        ax.set_xlabel(axLabels[0],fontsize=fontsize)
        ax.set_ylabel(axLabels[1],fontsize=fontsize)
        
    for spine in ax.spines.values():
        spine.set_color('k')
        spine.set_linewidth(0.5)

def groupScatterCorr(groups,data,labels=None,ax=None,correl='pearson',
                     groupStats=True,addOverall=False,fontsize=14,textOffset=0): 
    # "data" is two arrays or a two-column array, "labels" is two strings
    if ax is None:
        ax = plt.gca()
    
    if isinstance(data,np.ndarray): # columns into list
        data = [data[:,0],data[:,1]]
    
    gNames = groupNames(groups)
    for gInd,g in enumerate(gNames):
        scatterCorr(data[0][groups==g],data[1][groups==g],ax=ax,color=setGroupColor(g),gName=g,
                    correl=correl,line=groupStats,text=groupStats)
    #textOffset = 0.1
    if addOverall:
        gName = 'Combined' if groupStats else None
        scatterCorr(data[0],data[1],gName=gName,ax=ax,correl=correl,scatter=False)
        textOffset += 0.2

    def formatAxis(ax,xInput,yInput,labels,fontsize,offset=0.1,textOffset=0):
        # set axis labels, lims. run once per axis
        x,y = keepFinite(xInput,yInput)
        xOffset = (max(x) - min(x)) * offset
        yOffset = (max(y) - min(y)) * np.array([offset, offset + textOffset])
        ax.set_xlim(min(x)-xOffset,max(x)+xOffset)
        ax.set_ylim(min(y)-yOffset[0],max(y)+yOffset[1])
        if labels is not None:
            ax.set_xlabel(labels[0],fontsize=fontsize)
            ax.set_ylabel(labels[1],fontsize=fontsize)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize-4)
    formatAxis(ax,data[0],data[1],labels,fontsize,offset=0.1,textOffset=textOffset)
    plt.tight_layout()

