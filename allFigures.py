# -*- coding: utf-8 -*-
"""
Plots and saves all figures for PLOSOne Manuscript

@author: wolffbs
"""

import os
import matplotlib.pyplot as plt

# functions in current directory
import plotDaily
import circadianImgs
import arenaTests as arenas
import phenotyperTests as phenotyper
import bdnf as bdnf
#from . 

path = os.getcwd() + '/../figures/'
dpi=300
plt.close('all')

dailyFigsize = (5,4)
swarmFigsize = (2.5,4)
corrFigsize = (4,4)
perfFigsize = (5,4)
fontsize = 16

def fig2(): 
    # vwra
    fig,_ = plotDaily.standardPlots('cogVWRA',normalize=0,fontsize=fontsize,
                                    dailyFigsize=dailyFigsize,swarmFigsize=None)
    fig.savefig(path+'2A vwra daily.tif',dpi=dpi)
    
    _,fig = plotDaily.standardPlots('cogVWRA',normalize=7,fontsize=fontsize,
                                    dailyFigsize=None,swarmFigsize=swarmFigsize,dotsize=6,
                                    statistic='t',drawSig=True,shapiro=True)
    fig.savefig(path+'2B vwra box.tif',dpi=dpi)
    
    # ymaze
    figs,fig2 = arenas.yMazePlots(fontsize=fontsize,figsize=swarmFigsize)
    figs[0].savefig(path+'2C ymaze distance.tif',dpi=dpi)
    figs[1].savefig(path+'S1C ymaze arms.tif',dpi=dpi)
    figs[2].savefig(path+'2D ymaze perf.tif',dpi=dpi)
    fig2.savefig(path+'S1D ymaze time.tif',dpi=dpi)
    
    # open field
    figs,fig2 = arenas.openFieldPlots(fontsize=fontsize,figsize=swarmFigsize)
    figs[0].savefig(path+'2E ofield distance.tif',dpi=dpi)
    figs[1].savefig(path+'2F ofield center.tif',dpi=dpi)
    fig2.savefig(path+'S1E ofield time.tif',dpi=dpi)
    
    # correlations
    figs = arenas.vwraCorr(fontsize=fontsize,figsize=corrFigsize,groupStats=True)
    figs[0].savefig(path+'S1F vwra ymaze distance.tif',dpi=dpi)
    figs[1].savefig(path+'2G vwra ymaze perf.tif',dpi=dpi)
    figs[2].savefig(path+'S1G vwra ofield distance.tif',dpi=dpi)
    figs[3].savefig(path+'S1H vwra ofield center.tif',dpi=dpi) 

def fig3(): 
    # cogwall distance and food pellets
    fig,_ = plotDaily.standardPlots('allCogWall',removeDays=[0,1,2],normalize=0,fontsize=fontsize,
                                    plotLims=[-4,8],dailyFigsize=dailyFigsize,swarmFigsize=None)
    fig.savefig(path+'3A cogwall daily.tif',dpi=dpi)
    
    _,fig = plotDaily.standardPlots('allCogWall',normalize=1,fontsize=fontsize,
                                    dailyFigsize=None,swarmFigsize=swarmFigsize,dotsize=7,
                                    statistic='t',drawSig=True,shapiro=True)
    fig.savefig(path+'3B cogwall box.tif',dpi=dpi)

    _,fig = plotDaily.standardPlots('allCogWall',measure='pellet',removeDays=[0,1,2],normalize=0,
                                    dailyFigsize=None,swarmFigsize=swarmFigsize,dotsize=7,
                                    fontsize=fontsize,statistic='t',drawSig=True,shapiro=True)
    fig.savefig(path+'3C cogwall food.tif',dpi=dpi)
    
    # cogwall line plots
    figs = phenotyper.plotSegsGroupPerfPokesTime(segments=[2,4,5],figsize=perfFigsize,fontsize=fontsize)
    figs[0].savefig(path+'3E training time.tif',dpi=dpi)
    figs[1].savefig(path+'3H training pokes.tif',dpi=dpi)
    figs[2].savefig(path+'S2E retraining time.tif',dpi=dpi)
    figs[3].savefig(path+'S2F retraining pokes.tif',dpi=dpi)
    figs[4].savefig(path+'3F reversal time.tif',dpi=dpi)
    figs[5].savefig(path+'3I reversal pokes.tif',dpi=dpi)
    
    #cogwall performance boxplots
    fig = phenotyper.plotPerfFirstNight(figsize=swarmFigsize,fontsize=fontsize)
    fig.savefig(path+'3G perf time.tif',dpi=dpi)
    fig = phenotyper.plotPerfNumPokes(figsize=swarmFigsize,fontsize=fontsize)
    fig.savefig(path+'3J perf time.tif',dpi=dpi)
    
    #cogwall correlations
    figs = phenotyper.corrPerf() # correlations
    figs[0].savefig(path+'S2G corr dist perf.tif',dpi=dpi)
    figs[1].savefig(path+'S2H corr food perf.tif',dpi=dpi)
    figs[2].savefig(path+'3D corr dist food.tif',dpi=dpi)

def fig4(): 
    # bdnf levels
    figs = bdnf.plotLevels(normalize=False,figsize=swarmFigsize,fontsize=fontsize)
    figs[0].savefig(path+'4A mBDNF box.tif',dpi=dpi)
    figs[1].savefig(path+'4B proBDNF box.tif',dpi=dpi)  
    
    # bdnf correlations
    figs = bdnf.plotCorr(normalize=False,figsize=corrFigsize,fontsize=fontsize,correl='pearson')
    figs[0].savefig(path+'S3A corr ymaze mBDNF.tif',dpi=dpi)
    figs[1].savefig(path+'4C corr ymaze proBDNF.tif',dpi=dpi)
    figs[2].savefig(path+'S3B corr ofield mBDNF.tif',dpi=dpi)
    figs[3].savefig(path+'S3C corr ofield proBDNF.tif',dpi=dpi)
    figs[4].savefig(path+'S3D corr vwra mBDNF.tif',dpi=dpi)
    figs[5].savefig(path+'S3E corr vwra proBDNF.tif',dpi=dpi)    
 
def supp(): # circadian images
    figs = circadianImgs.circadianImgs('cogVWRA',plotDates=False,saveFigs=False,
                                   individualImgs=False,normPercent=95)
    figs[0].savefig(path+'S1A circadian vwra irrad.tif',dpi=dpi)
    figs[1].savefig(path+'S1B circadian vwra sham.tif',dpi=dpi)    
    
    figs = circadianImgs.circadianImgs('allCogWall',plotDates=False,saveFigs=False,
                                       individualImgs=False,normPercent=95)
    figs[0].savefig(path+'S2A circadian distance irrad.tif',dpi=dpi)
    figs[1].savefig(path+'S2B circadian distance sham.tif',dpi=dpi)
    
    figs = circadianImgs.circadianImgs('allCogWall',plotDates=False,saveFigs=False,
                                       individualImgs=False,normPercent=95,measure='pellet')
    figs[0].savefig(path+'S2C circadian pellet irrad.tif',dpi=dpi)
    figs[1].savefig(path+'S2D circadian pellet sham.tif',dpi=dpi) 

