#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle,sys,os,math
from itertools import groupby
from matplotlib.widgets import RadioButtons,Button,CheckButtons,TextBox
sys.path.insert(0, '../../programs/dataProcessing/')
from miscFunctions import reindexDataFrame,returnTicks
from matplotlib import colors,ticker
from scipy.signal import savgol_filter
from operator import itemgetter

def returnLogYTicksAndLabels(yvals):
    miny = math.floor(yvals)
    maxy = math.ceil(yvals)
    allyticks = list(range(miny,maxy+1))
    allyticklabels = []
    for ytick in allyticks:
        allyticklabels.append('$10^{'+str(ytick)+'}$')
    minoryticks = np.log10(list(range(2,10)))
    allminoryticks = []
    for ytick in allyticks[:-1]:
        for minory in minoryticks:
            allminoryticks.append(ytick+minory)
    return allyticks,allyticklabels,allminoryticks
        
def plot(plottingDf,subsettedDf,kwargs,facetKwargs,auxillaryKwargs,plotOptions):
    #Will need to update to make sure it pulls from y axis variable
    yvar = kwargs.pop('y')
    if auxillaryKwargs['subPlotType'] == 'histogram':
        fg = sns.FacetGrid(plottingDf,legend_out=True,**facetKwargs,**kwargs,**plotOptions['Y']['figureDimensions'])
        if plotOptions['Y']['axisScaling'] == 'Logarithmic':
            hist_kws = {'hist_kws':{'log':True}} 
        else:
            hist_kws = {}
        fg.map(sns.distplot,yvar,bins=256,kde=False,**hist_kws)
    elif auxillaryKwargs['subPlotType'] == 'kde':
        if auxillaryKwargs['dataType'] != 'singlecell':
            fg = sns.FacetGrid(plottingDf,**facetKwargs,**kwargs,**plotOptions['Y']['figureDimensions'])
            fg.map(sns.kdeplot,yvar,shade=False,bw=15)
        else:
            kwargIndices = []
            cols = list(plottingDf.columns)
            for kwarg in ['hue','row','col']:
                if kwarg in kwargs:
                    kwargIndices.append(cols.index(kwargs[kwarg]))
            kwargIndices = sorted(kwargIndices)
            uniqueKwargCombinations = [list(x) for x in set(tuple(x) for x in list(plottingDf.iloc[:,kwargIndices].values))]
            hist = [0]
            indexTuples = []
            for kwargCombo in uniqueKwargCombinations:
                selectDf = plottingDf.copy()
                for kwarg,kwargIndex in zip(kwargCombo,kwargIndices):
                    selectDf = selectDf[selectDf[list(selectDf.columns)[kwargIndex]] == kwarg]
                subplotValueList = selectDf['MFI'].values
                newvals = np.append(subplotValueList,[[0,1023]])
                temphist,_ = np.histogram(newvals, bins=256)
                temphist[0]-=1
                temphist[-1]-=1
                if auxillaryKwargs['plotspecifickwargs']['scaleToMode']:
                    temphist = [x/max(temphist) for x in temphist]
                hist+=list(temphist)
                for i in range(len(list(temphist))):
                    indexTuples.append(kwargCombo)

            hist = hist[1:]
            numUniquePlots = len(uniqueKwargCombinations)
            histBins = np.tile(list(range(1,1024,4)),numUniquePlots)
            smoothedHistBins = savgol_filter(hist, auxillaryKwargs['plotspecifickwargs']['smoothing']-1, 2) 
            
            if not auxillaryKwargs['plotspecifickwargs']['scaleToMode']:
                if plotOptions['Y']['axisScaling'] == 'Logarithmic':
                    cutoff = 1
                else:
                    cutoff = 0
            else:
                if plotOptions['Y']['axisScaling'] == 'Logarithmic':
                    cutoff = 0.001 
                else:
                    cutoff = 0
            for i,val in enumerate(smoothedHistBins):
                if val < cutoff:
                    smoothedHistBins[i] = cutoff
             
            mi  = pd.MultiIndex.from_tuples(indexTuples,names=itemgetter(*kwargIndices)(list(plottingDf.columns)))
            if auxillaryKwargs['plotspecifickwargs']['scaleToMode']:
                hist = [x*100 for x in hist]
                smoothedHistBins = [x*100 for x in smoothedHistBins]
                columns = ['MFI','% Max']
            else:
                columns = ['MFI','Count']

            #Construct correct array of data
            data = np.matrix([histBins,smoothedHistBins]).T

            newdf = pd.DataFrame(data,columns=columns,index=mi)
            plottingDf = newdf.reset_index()
            
            fg = sns.relplot(data=plottingDf,kind='line',x='MFI',y=columns[1],facet_kws=facetKwargs,**kwargs,**plotOptions['Y']['figureDimensions'])

            xtickValues,xtickLabels = returnTicks([-1000,1000,10000,100000])
            maxVal = max(subsettedDf.values)[0]
            minVal = min(subsettedDf.values)[0]
            if xtickValues[0] < minVal:
                minVal = xtickValues[0]
            if xtickValues[-1] > maxVal:
                maxVal = xtickValues[-1]
            for i,axis in enumerate(fg.axes.flat):
                axis.set_xticks(xtickValues)
                axis.set_xticklabels(xtickLabels)
                axis.set_xlim([minVal,maxVal])
                if plotOptions['Y']['axisScaling'] == 'Logarithmic':
                    axis.set_yscale('log')
            
    #fg.add_legend()
    return fg
