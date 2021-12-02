#!/usr/bin/env python3 
import json,pickle,math,matplotlib,sys,os,string,re
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
import itertools
from matplotlib import pyplot as plt
from collections import OrderedDict
from operator import itemgetter
sys.path.insert(0, '../dataprocessing/')
from miscFunctions import sortSINumerically,reindexDataFrame,setMaxWidth,returnSpecificExtensionFiles,returnTicks

def createParameterSelectionRadiobuttons(radiobuttonWindow,parameterList,parameterValueDict):
    radiobuttonList = []
    radiobuttonVarsDict = {}
    for i,parameter in enumerate(parameterList):
        tk.Label(radiobuttonWindow,text=parameter).grid(row=0,column=i,sticky=tk.W)
        parameterValues = parameterValueDict[parameter]
        parameterVar = tk.StringVar()
        for j in range(len(parameterValues)):
            parameterValue = parameterValues[j]
            radiobutton = tk.Radiobutton(radiobuttonWindow,text=parameterValue,value=parameterValue,variable=parameterVar)
            radiobutton.grid(row=j+1,column=i,sticky=tk.W)
        parameterVar.set(parameterValues[0])
        radiobuttonList.append(radiobutton)
        radiobuttonVarsDict[parameter] = parameterVar
    return radiobuttonList,radiobuttonVarsDict

def getRadiobuttonValues(radiobuttonVarsDict):
    parameterDict = {}
    for parameter in radiobuttonVarsDict:
        parameterDict[parameter] = radiobuttonVarsDict[parameter].get()
    return parameterDict

def createParameterAdjustmentSliders(sliderWindow,parameterList,parameterBoundsDict):
    sliderList = []
    for i,parameter in enumerate(parameterList):
        tk.Label(sliderWindow,text=parameter).grid(row=0,column=i)
        parameterBounds = parameterBoundsDict[parameter]
        slider = tk.Scale(sliderWindow, from_=parameterBounds[0], to=parameterBounds[1],resolution=parameterBounds[2])
        slider.grid(row=1,column=i)
        slider.set(parameterBounds[3])
        sliderList.append(slider)
    return sliderList
    
def getSliderValues(sliders,parameterList,mutuallyExclusiveParameterList = []):
    parametersForSliderFunction = {}
    for slider,parameter in zip(sliders,parameterList):
        parametersForSliderFunction[parameter] = slider.get()
    return parametersForSliderFunction

def createParameterSelectionDropdowns(dropdownWindow,parameterList,parameterValueDict,defaultParameterValueDict):
    dropdownList = []
    dropdownVarsDict = {}
    for i,parameter in enumerate(parameterList):
        parameterValues = parameterValueDict[parameter]
        tk.Label(dropdownWindow,text=parameter+': ').grid(row=i,column=0)
        parameterVar = tk.StringVar()
        parameterMenu = tk.OptionMenu(dropdownWindow,parameterVar,*parameterValues)
        parameterMenu.grid(row=i,column=1)
        parameterVar.set(defaultParameterValueDict[parameter])
        setMaxWidth(parameterValues,parameterMenu)
        dropdownList.append(parameterMenu)
        dropdownVarsDict[parameter] = parameterVar
    return dropdownList,dropdownVarsDict

def getDropdownValues(dropdownVarsDict):
    parametersForDropdowns = {}
    for parameter in dropdownVarsDict:
        parametersForDropdowns[parameter] = dropdownVarsDict[parameter].get()
    return parametersForDropdowns

def fixDuckTyping(plottingDf,kwargs):
    #Fix duck typing issue with replots: https://github.com/mwaskom/seaborn/issues/1653
    if 'hue' in kwargs and isinstance(plottingDf[kwargs['hue']][0],str):
        if plottingDf[kwargs['hue']][0].isnumeric():
            plottingDf[kwargs['hue']] = ["$%s$" % x for x in plottingDf[kwargs['hue']]]
    if 'size' in kwargs and isinstance(plottingDf[kwargs['size']][0],str):
        if plottingDf[kwargs['size']][0].isnumeric():
            plottingDf[kwargs['size']] = ["$%s$" % x for x in plottingDf[kwargs['size']]]
    return plottingDf

def get_cluster_centroids(plottingDf):
    clusterCentroids = []
    for cluster in pd.unique(plottingDf['Cluster']): 
        numeric = re.findall(r'\d+', cluster)
        clusterSubset = plottingDf[plottingDf['Cluster'] == cluster]
        clusterX = list(clusterSubset['Dimension 1'])
        clusterY = list(clusterSubset['Dimension 2'])
        clusterCentroid = (sum(clusterX) / len(clusterX), sum(clusterY) / len(clusterX))
        clusterCentroids.append([str(numeric[0]),clusterCentroid])
    return clusterCentroids

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def returnOrderedClusters(clusterList):
    numericClusters = []
    clusterDict = {}
    for cluster in clusterList:
        numeric = re.findall(r'\d+', cluster)
        clusterDict[numeric] = cluster
        numericClusters.append(numeric)
    orderedNumericClusterList = sorted(numericClusters)
    orderedClusters = []
    for orderedCluster in orderedNumericClusterList:
        orderedClusters.append(str(clusterDict[orderedCluster]))
    return orderedClusters

def returnOriginalOrders(trueLabelDict,plottingDf,kwargs,dimensionality):
    if dimensionality == '2d':
        newkwargs = kwargs.copy()
        a = newkwargs.pop('x')
        b = newkwargs.pop('y')
    else:
        newkwargs = kwargs.copy()
    orderDict = {}
    for kwarg in newkwargs:
        if newkwargs[kwarg] == 'Cluster' and '$' not in plottingDf[newkwargs[kwarg]][0]:
            orderedValues = list(map(str,sorted(list(map(int,list(pd.unique(plottingDf['Cluster'])))))))
        else:
            if newkwargs[kwarg] in trueLabelDict.keys():
                originalValues = trueLabelDict[newkwargs[kwarg]]
                newValues = pd.unique(plottingDf[newkwargs[kwarg]])
                orderedValues = []
                for originalValue in originalValues:
                    if originalValue in newValues:
                        orderedValues.append(originalValue)
            else:
                if len(pd.unique(plottingDf[newkwargs[kwarg]])) == 1:
                    orderedValues = [pd.unique(plottingDf[newkwargs[kwarg]])] 
                else:
                    orderedValues = list(pd.unique(plottingDf[newkwargs[kwarg]]))
        if newkwargs[kwarg] != 'None':
            if kwarg == 'x':  
                orderDict['order'] = orderedValues
            else:
                orderDict[kwarg+'_order'] = orderedValues
    return orderDict

def addLogicleXandCountYAxes(axis,subplotValues):
    xtickValues,xtickLabels = returnTicks([-1000,100,1000,10000,100000])
    maxVal = max(subplotValues)
    minVal = min(subplotValues)
    if  xtickValues[0] < minVal:
        minVal = xtickValues[0]
    if  xtickValues[-1] < maxVal:
        maxVal = xtickValues[-1]

    #Add in correct y axis labels
    maxCounts = []
    #Make sure bins of histogram are over the correct range by appending the appropriate extrema
    newvals = np.append(subplotValues,[[0,1023]])
    hist,_ = np.histogram(newvals, bins=256)
    #remove appended extrema
    hist[0]-=1
    hist[-1]-=1
    maxCount = max(hist)
    maxCounts.append(maxCount)
    #Add appropriate xtick values (also ytick values if kde) for each axis in figure
    axis.set_xticks(xtickValues)
    axis.set_xticklabels(xtickLabels)
    #axis.set_xlim([minVal,maxVal])
    oldylabels = axis.get_yticks().tolist()
    oldmax = oldylabels[-1]
    i=0
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

def updateDropdownControlledCompositionPlot(frameCanvas,plotAxis,plottingDf,trueLabelDict,levelVars,compositionParameter,legendoffset=1.7):
    plotAxis.clear()
    if not isinstance(plottingDf,list):
        parameters = [levelVars['x'].get(),levelVars['hue'].get()]
        parameters2 = ['x','hue']
        newkwargs = {}
        for parameter,parameter2 in zip(parameters,parameters2):
            if parameter != 'None':
                newkwargs[parameter2] = parameter
        #plottingDf = fixDuckTyping(plottingDf,newkwargs)
        orderedClusters = list(map(str,sorted(list(map(int,list(pd.unique(plottingDf['Cluster'])))))))
        if newkwargs['x'] in list(pd.unique(plottingDf['Feature'])):
            modifiedNewKwargs = newkwargs.copy()
            feature = modifiedNewKwargs.pop('x')
            featureBool = True
            newkwargs['x'] = 'Metric'
        else:
            modifiedNewKwargs = newkwargs.copy()
            featureBool = False
        orderDict = returnOriginalOrders(trueLabelDict,plottingDf,modifiedNewKwargs,'1.5d')
        if not featureBool:
            palette = sns.color_palette(sns.color_palette(),len(pd.unique(plottingDf['Cluster'])))
        else:
            palette = sns.color_palette(sns.color_palette(),len(pd.unique(plottingDf[newkwargs['hue']])))
        kdePlotBool = False
        if compositionParameter == 'frequency':
            if is_number(plottingDf[newkwargs['x']][0]) and len(pd.unique(plottingDf[newkwargs['x']])) > 4 and newkwargs['x'] != 'Cluster':
                if not featureBool:
                    #"Unstack" dataframe
                    plottingDf = plottingDf[plottingDf['Feature'] == plottingDf['Feature'][0]]
                    for i,cluster in enumerate(orderedClusters):
                        g3 = sns.kdeplot(plottingDf[plottingDf['Cluster'] == cluster][newkwargs['x']],color=palette[i],ax=plotAxis,shade=True,label=cluster)
                else:
                    plottingDf = plottingDf[plottingDf['Feature'] == feature]
                    if newkwargs['hue'] == 'Cluster':
                        hue_order = orderedClusters 
                    else:
                        hue_order = orderDict['hue_order'] 
                    for i,cluster in enumerate(hue_order):
                        g3 = sns.kdeplot(plottingDf[plottingDf[newkwargs['hue']] == cluster][newkwargs['x']],color=palette[i],ax=plotAxis,shade=True,label=cluster)
                    if max(plottingDf['Metric']) > 100:
                        addLogicleXandCountYAxes(plotAxis,plottingDf['Metric'])
                        #xticksToUse = [-1000,100,1000,10000,100000]
                        #xtickValues,xtickLabels = returnTicks(xticksToUse)
                        #plotAxis.set_xticks(xtickValues)
                        #plotAxis.set_xticklabels(xtickLabels)
                kdePlotBool = True
            else:
                #"Unstack" dataframe
                plottingDf = plottingDf[plottingDf['Feature'] == plottingDf['Feature'][0]]
                g3 = sns.countplot(data=plottingDf,ax=plotAxis,**newkwargs,**orderDict,edgecolor='black',linewidth=1)
        else: 
            #"Unstack" dataframe
            plottingDf = plottingDf[plottingDf['Feature'] == plottingDf['Feature'][0]]
            if 'hue' in newkwargs.keys():
                x,y = newkwargs['x'],newkwargs['hue']
                plottingDf = plottingDf.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()
            else:
                plottingDf = plottingDf[newkwargs['x']].value_counts(normalize=True).mul(100)
                plottingDf.index.names = [newkwargs['x']]
                plottingDf = plottingDf.to_frame('percent').reset_index()
            newkwargs['y'] = 'percent'
            g3 = sns.barplot(data=plottingDf,ax=plotAxis,**newkwargs,**orderDict,edgecolor='black',linewidth=1)
    
        #Rotate index labels if x not cluster
        if newkwargs['x'] != 'Cluster':
            #if not kdeplot, rotate index labels
            if not kdePlotBool:
                plt.setp(plotAxis.xaxis.get_majorticklabels(), rotation=45)
            #If feature plot on x axis, change x axis name
            if featureBool:
                plotAxis.set(xlabel=feature)
        else:
            #If count plot, color
            if not kdePlotBool:
                for i,cluster in enumerate(orderedClusters):
                    plotAxis.get_xticklabels()[i].set_color(palette[i])
            plotAxis.set(xlabel='Group')
        #Correct legend for kde plot
        if kdePlotBool:
            if newkwargs['hue'] == 'Cluster':
                newkwargs['hue'] = 'Group'
            leg = g3.legend(loc='center right', bbox_to_anchor=(legendoffset, 0.5), ncol=1,framealpha=0,title=newkwargs['hue'])
        else:
            leg = g3.legend(loc='center right', bbox_to_anchor=(legendoffset, 0.5), ncol=1,framealpha=0)

    frameCanvas.draw()

def updateDropdownControlledPlot(frameCanvas,plotAxis,plottingDf,levelVars,xColumn,yColumn,alpha=0.3,legendoffset=-0.1,trueLabelDict = []):
    plotAxis.clear()
    
    parameters = [levelVars['hue'].get(),levelVars['style'].get(),levelVars['size'].get()]
    parameters2 = ['hue','style','size']
    newkwargs = {'x':xColumn,'y':yColumn}
    numLegendElements = 0
    for parameter,parameter2 in zip(parameters,parameters2):
        if parameter != 'None':
            newkwargs[parameter2] = parameter
            val = plottingDf[parameter][0]
            if isinstance(val,(int,float,np.integer)):
                numLegendElements+=4
            else:
                numLegendElements+=len(pd.unique(plottingDf[parameter]))
    if not isinstance(trueLabelDict,list):
        orderDict = returnOriginalOrders(trueLabelDict,plottingDf,newkwargs,'2d')
    else:
        orderDict = {}
    plottingDf = fixDuckTyping(plottingDf,newkwargs)
    if 'hue' in newkwargs.keys():
        if not isinstance(plottingDf[newkwargs['hue']][0],str):
            print('COOLWARM')
            palette = 'coolwarm'
            newkwargs['palette'] = palette
        else:
            print('NONE')
            palette = sns.color_palette(sns.color_palette(),len(pd.unique(plottingDf[newkwargs['hue']]))) 
            newkwargs['palette'] = palette
    if 'hue' in newkwargs.keys():
        if newkwargs['hue'] == 'Cluster':
            clusters = list(pd.unique(plottingDf['Cluster']))
            tempDict = {}
            for i in range(len(clusters)):
                cluster = clusters[i]
                numeric = re.findall(r'\d+', cluster)
                tempDict[int(numeric[0])] = cluster
            sortedTempDict = OrderedDict(sorted(tempDict.items()))
            orderDict['hue_order'] = list(sortedTempDict.values())
    if 'Event' in plottingDf.columns:
        g3 = sns.scatterplot(data=plottingDf,ax=plotAxis,alpha=0.7,s=3,**newkwargs,**orderDict)
    else:
        g3 = sns.scatterplot(data=plottingDf,ax=plotAxis,alpha=0.7,**newkwargs,**orderDict)
    if 'hue' in newkwargs.keys():
        if newkwargs['hue'] == 'Cluster':
            clusterCentroids = get_cluster_centroids(plottingDf)
            for i in range(len(clusterCentroids)):
                g3.annotate(clusterCentroids[i][0],xy=clusterCentroids[i][1])
    legendSpillover = 15
    leg = g3.legend(loc='center right', bbox_to_anchor=(legendoffset, 0.5), ncol=math.ceil(numLegendElements/legendSpillover),framealpha=0)
    for t in leg.texts:
        # truncate label text to 4 characters
        if t.get_text() == '1.2000000000000002':
            t.set_text('1.0')
        else:
            if '.' in t.get_text():
                if t.get_text().replace('.','',1).isdigit():
                    decimalIndex = t.get_text().find('.')
                    t.set_text(round(float(t.get_text()),2))
    frameCanvas.draw()

#Assign level with largest num unique values to hue, 2nd largest to style, time to size to start
def getDefaultKwargs(df):
    kwargs = {}
    responseColumns = []
    numUniqueElements = []
    tempDict = {}
    for column in df.columns:
        if 'Dimension' not in column and 'Time' not in column and 'Replicate' not in column:
            responseColumns.append(column)
            numUniqueElements.append(len(list(pd.unique(df[column]))))
            tempDict[column] = len(list(pd.unique(df[column])))
    columnsLeftToAssign = responseColumns.copy()
    sortedNumUniqueElements = sorted(numUniqueElements)[::-1] 
    numUniqueElements2 = numUniqueElements.copy()
    sortedNumUniqueElements2 = sortedNumUniqueElements.copy()
    
    sortedTempDict = OrderedDict(sorted(tempDict.items(), key=itemgetter(1),reverse=True))
    #sortedTempDict = sorted(tempDict, key=tempDict.get)

    #Assign hue variable
    maxUniqueElementsColumn = responseColumns[numUniqueElements.index(sortedNumUniqueElements[0])]
    kwargs['hue'] = maxUniqueElementsColumn
    
    columnsLeftToAssign.remove(maxUniqueElementsColumn)
    numUniqueElements2.remove(sortedNumUniqueElements2[0])
    sortedNumUniqueElements2 = sortedNumUniqueElements2[1:]
    
    if len(sortedNumUniqueElements) > 1:
        #Assign style variable
        if sortedNumUniqueElements[1] < 7:
            secondMaxUniqueElementsColumn = list(sortedTempDict.keys())[1]
            #secondMaxUniqueElementsColumn = responseColumns[numUniqueElements.index(sortedNumUniqueElements[1])]
            print(responseColumns[numUniqueElements.index(sortedNumUniqueElements[1])])
            kwargs['style'] = secondMaxUniqueElementsColumn
            columnsLeftToAssign.remove(secondMaxUniqueElementsColumn)
            numUniqueElements2.remove(sortedNumUniqueElements2[0])
            sortedNumUniqueElements2 = sortedNumUniqueElements2[1:]
    
    if len(sortedNumUniqueElements) > 2:
        #Assign size variable
        if len(columnsLeftToAssign) > 0:
            if sortedNumUniqueElements2[0] > 1:
                thirdMaxUniqueElementsColumn = columnsLeftToAssign[numUniqueElements2.index(sortedNumUniqueElements2[0])]
                kwargs['size'] = thirdMaxUniqueElementsColumn
            else:
                if 'Time' in df.columns and len(pd.unique(df['Time'])) > 1:
                    kwargs['size'] = 'Time'
        else:
            if 'Time' in df.columns and len(pd.unique(df['Time'])) > 1:
                kwargs['size'] = 'Time'
    
    defaultDict = {'hue':'None','style':'None','size':'None'}
    defaultplotkwargs = kwargs.copy()
    if 'Event' not in df.columns:
        if 'hue' in defaultplotkwargs.keys():
            defaultDict['hue'] = defaultplotkwargs['hue']
        if 'style' in defaultplotkwargs.keys():
            defaultDict['style'] = defaultplotkwargs['style']
        if 'size' in defaultplotkwargs.keys():
            defaultDict['size'] = defaultplotkwargs['size']
    return kwargs,defaultDict
