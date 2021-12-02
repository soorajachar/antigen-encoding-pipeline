#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle,sys,os
from itertools import groupby
from matplotlib.widgets import RadioButtons,Button,CheckButtons,TextBox
sys.path.insert(0, '../../programs/dataProcessing/')
from miscFunctions import reindexDataFrame,returnTicks
from matplotlib import colors,ticker
        
def plot(plottingDf,subsettedDf,kwargs,facetKwargs,auxillaryKwargs,plotOptions):
    whitespace = 0.8
    #Will need to update to make sure it pulls from y axis variable
    yvar = kwargs.pop('y')
    if auxillaryKwargs['subPlotType'] == 'kde':
        if auxillaryKwargs['dataType'] == 'singlecell':
            facetKwargs['sharey'] = 'none' 
        fg = sns.FacetGrid(plottingDf,**facetKwargs,**kwargs,**plotOptions['Y']['figureDimensions'])
        #fg.map(sns.kdeplot,yvar,shade=False)
        fg.map(sns.kdeplot,yvar,shade=False,bw=15)
    elif auxillaryKwargs['subPlotType'] == 'histogram':
        fg = sns.FacetGrid(plottingDf,legend_out=True,**facetKwargs,**kwargs,aspect=2,**plotOptions['X']['figureDimensions'])
        fg.map(sns.distplot,yvar,bins=256)#,kde=False)
    if auxillaryKwargs['dataType'] == 'singlecell':
        xtickValues,xtickLabels = returnTicks([-1000,1000,10000,100000])
        maxVal = max(subsettedDf.values)[0]
        minVal = min(subsettedDf.values)[0]
        if  xtickValues[0] < minVal:
            minVal = xtickValues[0]
        if  xtickValues[-1] < maxVal:
            maxVal = xtickValues[-1]

        #Add in correct y axis labels
        maxCounts = []
        subplotValuesList = []
        if 'row' in kwargs and 'col' in kwargs:
            for rowVal in pd.unique(plottingDf[kwargs['row']]):
                for colVal in pd.unique(plottingDf[kwargs['col']]):
                    subplotValues = plottingDf[(plottingDf[kwargs['row']] == rowVal) & (plottingDf[kwargs['col']] == colVal)][yvar]
                    subplotValuesList.append(subplotValues)
        else:
            if 'row' in kwargs:
                for rowVal in pd.unique(plottingDf[kwargs['row']]):
                    subplotValues = plottingDf[plottingDf[kwargs['row']] == rowVal][yvar]
                    subplotValuesList.append(subplotValues)
            elif 'col' in kwargs:
                for colVal in pd.unique(plottingDf[kwargs['col']]):
                    subplotValues = plottingDf[plottingDf[kwargs['col']] == colVal][yvar]
                    subplotValuesList.append(subplotValues)
            else:
                subplotValuesList = [plottingDf[yvar]]

        for subplotValues in subplotValuesList: 
            #Make sure bins of histogram are over the correct range by appending the appropriate extrema
            newvals = np.append(subplotValues,[[0,1023]])
            hist,_ = np.histogram(newvals, bins=256)
            #remove appended extrema
            hist[0]-=1
            hist[-1]-=1
            maxCount = max(hist)
            maxCounts.append(maxCount)
        #Add appropriate xtick values (also ytick values if kde) for each axis in figure
        for i,axis in enumerate(fg.axes.flat):
            axis.set_xticks(xtickValues)
            axis.set_xticklabels(xtickLabels)
            axis.set_xlim([minVal,maxVal])
            oldylabels = axis.get_yticks().tolist()
            oldmax = oldylabels[-1]
            
            minticknum = 5
            factor = 10
            keepIncreasing = True
            while keepIncreasing:
                tickspaces = [1*factor,2*factor,2.5*factor,5*factor,10*factor]
                print(tickspaces)
                for j,tickspace in enumerate(tickspaces):
                    numticks = int(maxCounts[i]/tickspace)
                    #uncomment if you want the min tick number to be "minticknumber"
                    #if numticks <= minticknum:
                    #uncomment if you want the max tick number to be "minticknumber"
                    if numticks <= minticknum:
                        if j == 0:
                            finalTickLength = tickspaces[0]
                        else:
                            #uncomment if you want the max tick number to be "minticknumber"
                            finalTickLength = tickspaces[j]
                            #uncomment if you want the min tick number to be "minticknumber"
                            #finalTickLength = tickspaces[j-1]
                        keepIncreasing = False
                        break
                factor*=10
            
            print(maxCounts[i])
            if maxCounts[i] > 0:
                finalNumticks = int(maxCounts[i]/finalTickLength)+1
                if finalNumticks <= 1:
                    finalNumticks = 2
                oldTickLength = oldmax/(finalNumticks-1)
                newyticklabels = []
                newyticks = []
                for i in range(finalNumticks):
                    newyticks.append(i*oldTickLength)
                    newyticklabels.append(int(i*finalTickLength))
                axis.set_yticks(newyticks)
                axis.set_yticklabels(newyticklabels)
    
    fg.add_legend()
    return fg
