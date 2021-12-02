#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle,sys,os
from itertools import groupby
from matplotlib.widgets import RadioButtons,Button,CheckButtons,TextBox
sys.path.insert(0, '../../programs/dataProcessing/')
from miscFunctions import reindexDataFrame
from matplotlib import colors,ticker
    
def plot(plottingDf,subsettedDf,kwargs,facetKwargs,auxillaryKwargs,plotOptions):
    #Make sure there are markers at each column variable
    if 'Time' in plottingDf.columns:
        if len(pd.unique(plottingDf.Time)) > 36:
            if kwargs['x'] == 'Time' or kwargs['y'] == 'Time':
                print('wat1')
                fg = sns.relplot(data=plottingDf,kind=auxillaryKwargs['subPlotType'],facet_kws=facetKwargs,ci=False,**kwargs,**plotOptions['X']['figureDimensions'])
            else:
                print('wat2')
                fg = sns.relplot(data=plottingDf,kind=auxillaryKwargs['subPlotType'],facet_kws=facetKwargs,ci=False,**kwargs,**plotOptions['X']['figureDimensions'],sort=False,mew=0,ms=4)
        else:
            if 'style' not in kwargs.keys():
                print('wat3')
                fg = sns.relplot(data=plottingDf,marker='o',kind=auxillaryKwargs['subPlotType'],facet_kws=facetKwargs,ci=False,**kwargs,**plotOptions['X']['figureDimensions'],sort=False,mew=0,ms=4)
            else:
                print('wat4')
                fg = sns.relplot(data=plottingDf,markers=True,kind=auxillaryKwargs['subPlotType'],facet_kws=facetKwargs,ci=False,**kwargs,**plotOptions['X']['figureDimensions'],sort=False,mew=0,ms=4)
    else:
        if 'style' not in kwargs.keys() and auxillaryKwargs['subPlotType'] == 'line':
            print('wat5')
            fg = sns.relplot(data=plottingDf,marker='o',kind=auxillaryKwargs['subPlotType'],facet_kws=facetKwargs,ci=False,**kwargs,**plotOptions['X']['figureDimensions'],sort=False,mew=0,ms=4)
        else:
            if auxillaryKwargs['subPlotType']=='line':
                print('wat6')
                fg = sns.relplot(data=plottingDf,markers=True,kind=auxillaryKwargs['subPlotType'],facet_kws=facetKwargs,ci=False,**kwargs,**plotOptions['X']['figureDimensions'],sort=False,mew=0,ms=4)
            else:
                print('wat7')
                fg = sns.relplot(data=plottingDf,kind=auxillaryKwargs['subPlotType'],facet_kws=facetKwargs,**kwargs,**plotOptions['X']['figureDimensions'])
    #X and Y Axis Scaling for 2D plots
    for axis in plotOptions:
        k = len(fg.fig.get_axes())
        if 'Y' in axis:
            if plotOptions[axis]['axisScaling'] == 'Logarithmic':
                for i in range(k):
                    fg.fig.get_axes()[i].set_yscale('log')
            elif plotOptions[axis]['axisScaling'] == 'Biexponential':
                for i in range(k):
                    fg.fig.get_axes()[i].set_yscale('symlog',linthreshx=plotOptions[axis]['linThreshold'])
            
            if str(plotOptions[axis]['limit'][0]) != '' or str(plotOptions[axis]['limit'][1]) != '':
                for i in range(k):
                    if str(plotOptions[axis]['limit'][0]) != '' and str(plotOptions[axis]['limit'][1]) != '':
                        fg.fig.get_axes()[i].set_ylim(bottom=float(plotOptions[axis]['limit'][0]),top=float(plotOptions[axis]['limit'][1]))
                    else:
                        if str(plotOptions[axis]['limit'][0]) != '':
                            fg.fig.get_axes()[i].set_ylim(bottom=float(plotOptions[axis]['limit'][0]))
                        else:
                            fg.fig.get_axes()[i].set_ylim(top=float(plotOptions[axis]['limit'][1]))
        else:
            if plotOptions[axis]['axisScaling'] == 'Logarithmic':
                for i in range(k):
                    fg.fig.get_axes()[i].set_xscale('log')
            elif plotOptions[axis]['axisScaling'] == 'Biexponential':
                for i in range(k):
                    fg.fig.get_axes()[i].set_xscale('symlog',linthreshx=plotOptions[axis]['linThreshold']) 
            
            if str(plotOptions[axis]['limit'][0]) != '' or str(plotOptions[axis]['limit'][1]) != '':
                for i in range(k):
                    if str(plotOptions[axis]['limit'][0]) != '' and str(plotOptions[axis]['limit'][1]) != '':
                        fg.fig.get_axes()[i].set_xlim(bottom=float(plotOptions[axis]['limit'][0]),top=float(plotOptions[axis]['limit'][1]))
                    else:
                        if str(plotOptions[axis]['limit'][0]) != '':
                            fg.fig.get_axes()[i].set_xlim(bottom=float(plotOptions[axis]['limit'][0]))
                        else:
                            fg.fig.get_axes()[i].set_xlim(top=float(plotOptions[axis]['limit'][1])) 
    return fg
