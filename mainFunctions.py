# -*- coding: utf-8 -*-
"""
@author: Brian
"""
import numpy as np
import pandas as pd
import json 
import os
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def getMainPath():
	return os.getcwd() + '/data/'
    #/Users/sbruinsma/Desktop/Other Projects/ULAB/ulab_group_code

#%% defaults and path stuff
exptTypeDirs = {'VWRA':'VWRA',
                'cogWall':'phenotyper'}

distMeasures = ['distance','activity','speed','actTime','actSum']

measureDict = {'distance': 'Distance', # phenotyper or vwra
               'activity': 'Activity', # vwra
               'speed': 'Speed', # vwra
               'actTime': 'Time Active', # vwra
               'pellet': 'Food Pellets', # phenotyper
               'poke': 'Pokes'} # phenotyper
    
def getExptType(expt,exptType=None):
    if exptType is not None:
        return exptType
    for eType,eDir in exptTypeDirs.items():
        expts = os.listdir(getMainPath() + eDir)
        if expt.lower() in [e.lower() for e in expts]:
            return eType
    assert False, 'getExptType could not find directory for expt %s'%expt
    
def getDefaultMeasure(exptType):
    return 'actTime' if exptType == 'VWRA' else 'distance'

def getExptTypeAndMeasure(expt,exptType=None,measure=None):
    if exptType is None:
        exptType = getExptType(expt)
    if measure is None:
        measure = getDefaultMeasure(exptType)
    return exptType,measure

def getExptPath(expt,exptType=None):
    if exptType is None:
        exptType = getExptType(expt)
    return '%s%s/%s/'%(getMainPath(),exptTypeDirs[exptType],expt)

def setResetDays(expt):
	return [0]

def setResetLen(expt):
	return 3 # three days of irradiation

def measure2dType(measure):
    assert measure in measureDict.keys(), 'invalid measure: %s'%(measure)
    return measureDict[measure]
        
def updatePlotParamsFromFile(expt,plotParams,exptType=None,display=False):
    exptType = getExptType(expt)

    # update plotParams dict with any overwrites indicated in plotParams.txt file
    plotParamsFile = '%splotParams.txt'%getExptPath(expt,exptType)
    if not os.path.isfile(plotParamsFile):
        return plotParams # if no file, then don't update
    
    if display:
        print('\tUpdating plotParams from "%s/plotParams.txt"'%expt)
    
    try:
        with open(plotParamsFile) as file:
            newParams = json.load(file)
    except ValueError: 
        with open(plotParamsFile) as file:
            fLines = [l.strip().split('=') for l in file if l[0]!=';' and l[0]!='#']
            for fL in fLines:
                if len(fL) == 2:
                    exec('plotParams["%s"] = %s'%(fL[0].strip(),fL[1].strip()))
    else:
        plotParams = {**plotParams,**newParams} # update dictionary
    return plotParams    

def getPlotParams(expt=None,exptType=None,measure='distance',display=True):
    exptType,measure = getExptTypeAndMeasure(expt,exptType,measure)
    plotParams = {}
    
    ########################
    # set universal defaults
    plotParams['normalize'] = 0 # number of baseline days to use for normalization; set to zero to not normalize
    plotParams['groupIrradSham'] = False # set to None to leave group names alone
                # set to 'few' to group only pure irrad and sham and throw everything else out
                # set to 'all' to group everything into two groups (i.e. ignore treatments)
    plotParams['actTimeThresh'] = 0 # note that if 0, then phenotyper data will be very active
    plotParams['errType'] = 'stdErr' # 'stdErr' or 'CI'

    plotParams['removeDays'] = [] # remove these days from analysis
    plotParams['removeGroups'] = [] # remove this group from plots
    plotParams['removeIDs'] = [] # remove this animal ID from analysis
    plotParams['removeExpts'] = [] # remove this experiment from analysis
    
    plotParams['plotLims'] = None # plot from day plotLims[0] to day plotLims[1] (inclusive)

    ###################################
    # set defaults specific to exptType
    plotParams['dataType'] = measure2dType(measure) # measure string formatted for printing on plots
    
    # Units to write on figure legends
    if measure == 'distance':
        if exptType == 'VWRA':
            plotParams['units'] = 'm'
        elif exptType in ['phenotyper','cogWall']:
            plotParams['units'] = 'cm'
    elif measure == 'actTime':
        plotParams['units'] = 'min'
    elif measure == 'speed':
        if exptType == 'VWRA':
            plotParams['units'] = 'm/min'
        elif exptType in ['phenotyper','cogWall']:
            plotParams['units'] = 'cm/min'
    elif measure == 'tempMed':
        plotParams['units'] = u'\N{DEGREE SIGN}C'
    elif measure == 'hrMean':
        plotParams['units'] = 'bpm'
    else:
        plotParams['units'] = None
        
    plotParams = updatePlotParamsFromFile(expt,plotParams,exptType,display)
    
    ################################
    # set defaults specific for expt
    plotParams['resetDays'] = setResetDays(expt)
    plotParams['resetLen'] = setResetLen(expt)
    return plotParams


#####################################################################
#%% reading data
    
def daysFromDatetimes(dts,resetDate=None):
    if resetDate is None:
        resetDate = (dts.iloc[0] - pd.Timedelta(hours=6)).dt.date
    adts = (dts - pd.Timedelta(hours=6)).dt.date
    days = (adts - resetDate).dt.days
    return days

def getDataFile(expt,exptType=None,measure='distance'):
    exptPath = getExptPath(expt,exptType)
    m = 'distance' if measure in distMeasures else measure
    dataFiles = [f for f in os.listdir(exptPath) if f[-4:] == '.txt' and m in f]
    assert len(dataFiles), '%s data file not found in %s'%(m,exptPath)
    if len(dataFiles) > 1:
        dataFiles = [f for f in dataFiles if 'all' in f] # vwra can have "all" and "daily" files
        assert len(dataFiles) == 1, 'cannot identify unique %s data file in %s'%(m,exptPath)
    return exptPath + dataFiles[0]

def fixRowRepeats(data,function='mean',display=False): 
    g = data.groupby(data.index) # groupby removes multiindex
    if display:
        print('removing %d repeated rows:'%sum(g.size()>0))
        for d in g.size()[g.size() > 1].index.values:
            print('\t%s'%str(d[1]))
    data2 = getattr(g,function)()
    data2.index = pd.MultiIndex.from_tuples(data2.index).set_names(['Day','DateTime'])
    return data2

def readExptData(expt,exptType=None,measure='distance',fixRepeats=True):
    dataFile = getDataFile(expt,exptType,measure)
    data = pd.read_csv(dataFile,sep='\t',parse_dates=[['Date','Time']])
    data.index = pd.MultiIndex.from_frame(data[['Day','Date_Time']]).set_names(['Day','DateTime'])
    data.drop(columns=['Date_Time','Day'],inplace=True)
    if fixRepeats:
        data = fixRowRepeats(data)
    return data

# functions for extracting axis information after readExptData()
def getExpts(df):
    return np.array([c.split('-')[0] for c in df.columns])
def getIDs(df):
    return np.array([c.split(':')[0] for c in df.columns])    
def getGroups(df):
    if isinstance(df,pd.Series):
        return np.array([c.split(':')[1] for c in df.index])
    else:
        return np.array([c.split(':')[1] for c in df.columns])
    
def getDays(df):
    return np.array([i[0] for i in df.index])
def getDates(df): # returns pd.Series of datetime or date values with days (int) as index,
    days = df.index.get_level_values('Day')
    try: 
        return pd.Series(index=days,data=df.index.get_level_values('DateTime'))
    except KeyError:
        return pd.Series(index=days,data=df.index.get_level_values('Date'))
        
def getUniqueDates(df,hour=18): # returns just the dates from (day,datetime) multiindex
    dates = getDates(df)
    return dates[dates.dt.hour == hour].dt.date.drop_duplicates().rename('Date')

def getDateZero(df): # returns the date for which the day would be zero
    dates = getUniqueDates(df)
    dates0 = dates - pd.to_timedelta(dates.index,unit='days') # PerformanceWarning: not vectorized
    assert all(dates0 == dates0.iloc[0]), 'mismatch between days and dates'
    return dates0.iloc[0]

######################################################################
#%% data organization

# remove specified mice from data, plus some invalid values        
def removeDataMice(data,expts=[],groups=[],aIDs=[],display=False,
                   dropna=0.1, # higher dropna => accept fewer nans
                   invalidGroups = ['None',''],invalidIDs=[0,'0','00','0:','']):
    groups = np.concatenate((groups,invalidGroups))
    aIDs = np.concatenate((aIDs,invalidIDs))
    
    def removeData(data,cols,rem,display=False):
        remove = np.in1d(cols,rem)
        if display and sum(remove):
            print('\tRemoving %d mice...'%sum(remove))
        return data.loc[:,~remove]
    df = removeData(data,getExpts(data),expts,display)
    df = removeData(df,getGroups(df),groups,display)
    df = removeData(df,getIDs(df),aIDs,display)
    df.dropna(axis=1,thresh=len(df)*dropna)
    return df

def removeDataDays(data,days):
    return data.loc[~np.in1d(getDays(data),days),:]

def applyPlotLims(data,plotLims):
    if plotLims not in [None,[]]:
        data = data.loc[plotLims[0]:plotLims[1]]
    return data

# shift datetimes so that the irradDate is date0 is e.g. June 1, 1950 but times are the same
# useful for combining experiments run on different dates
def zeroDates(df,date0=pd.Timestamp('1950-06-01').date()):
    datesAdj = getDates(df) - (getDateZero(df) - date0)
    df.index = pd.MultiIndex.from_frame(datesAdj.reset_index())

