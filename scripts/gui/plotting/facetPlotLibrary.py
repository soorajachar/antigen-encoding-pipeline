#!/usr/bin/env python3
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os,json,pickle,math,itertools
from scipy import stats
import subprocess
from matplotlib.colors import LogNorm,SymLogNorm
sys.path.insert(0, 'scripts/gui/plotting/')
from miscFunctions import sortSINumerically,reindexDataFrame
from miscFunctions import returnGates,returnTicks
from operator import itemgetter
import facetPlot1D as fp1D
import facetPlotCategorical as fpCategorical
import facetPlot2D as fp2D
import facetPlot3D as fp3D
from facetPlot3D import draw_faceted_heatmap,returnHeatmapAspectRatios

idx = pd.IndexSlice
splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('antigen-encoding-pipeline')+1])+'/'

def produceSubsettedDataFrames(fulldf,withinFigureBoolean,specificValueBooleanList,trueLabelDict):
    print(withinFigureBoolean)
    print(specificValueBooleanList)
    #Get all possible subsetted indices
    figureSubsettedLevelValues = []
    withinFigureSubsettedLevelValues = []
    figureSubsettingLevels = []
    figureLevelNames = []
    figureLevelIndices = []
    levelValuesPlottedIndividually = []
    print(fulldf)
    if (len(fulldf.columns) > 1 and 'Node' in fulldf.columns[0]) or 'Feature' in fulldf.index.names:
        exclusionLevel = 'Time'
    else:
        exclusionLevel = 'Event'
    print(exclusionLevel)
    print('waaaaaat')
    print(fulldf)
    print(fulldf.columns.name)
    print(fulldf.columns[0])
    for levelIndex,currentLevelName in enumerate(fulldf.index.names):
        #Event level always has every value included (single cell)
        #Otherwise, check which level values in the level were selected by the user
        if currentLevelName != exclusionLevel:
            #currentLevelValues = pd.unique(fulldf.index.get_level_values(currentLevelName))
            currentLevelValues = trueLabelDict[currentLevelName]
            levelValues = []
            if ((currentLevelName == 'Marker' or currentLevelName == 'Markers') and exclusionLevel in fulldf.index.names):
                realLevelIndex = levelIndex - 1
            else:
                realLevelIndex = levelIndex
            #Go through each level value in the level; regardless of figure selection status, and add based on level value selection status
            for levelValue,specificBoolean in zip(currentLevelValues,specificValueBooleanList[realLevelIndex]):
                if specificBoolean:
                    levelValues.append(levelValue)
            print(levelValues)
            #If we will include this level within the figure
            if withinFigureBoolean[realLevelIndex]:
                withinFigureSubsettedLevelValues.append(levelValues)
                if len(levelValues) == len(currentLevelValues):
                    for levelValue in levelValues:
                        levelValuesPlottedIndividually.append(levelValue)
            #Only need to add level values to figure list; will be xs'd out of the full dataframe in the subsetted, within figure dataframes
            else:
                print('wat')
                figureSubsettedLevelValues.append(levelValues)
                figureLevelNames.append(currentLevelName)
                figureLevelIndices.append(levelIndex)
    if len(figureLevelIndices) > 0:
        #Get all row level values present in the dataframe
        allPossibleSubsettingCombos = itertools.product(*figureSubsettedLevelValues)
        if exclusionLevel not in fulldf.index.names:
            rowList = []
            for row in range(fulldf.shape[0]):
                allCurrentLevelValues = fulldf.iloc[row,:].name
                currentLevelValues = itemgetter(*figureLevelIndices)(allCurrentLevelValues)
                if not isinstance(currentLevelValues,tuple):
                    currentLevelValues = tuple([currentLevelValues])
                rowList.append(currentLevelValues)
            actualSubsettingCombos = []
            #From original dataframe; select all rows that appear in the all possible combination list
            for subsettingCombo in allPossibleSubsettingCombos:
                if subsettingCombo in rowList:
                    actualSubsettingCombos.append(subsettingCombo)
        else:
            rowList = []
            nonEventLevelList = []
            for level in fulldf.index.names:
                if level != exclusionLevel:
                    nonEventLevelList.append(level)
            noEventDf = fulldf.groupby(nonEventLevelList).first()
            for row in range(noEventDf.shape[0]):
                allCurrentLevelValues = noEventDf.iloc[row,:].name
                figureLevelIndices = [max(min(x, len(fulldf.index.names)-2), 0) for x in figureLevelIndices]
                currentLevelValues = itemgetter(*figureLevelIndices)(allCurrentLevelValues)
                if not isinstance(currentLevelValues,tuple):
                    currentLevelValues = tuple([currentLevelValues])
                rowList.append(currentLevelValues)
            actualSubsettingCombos = []
            #From original dataframe; select all rows that appear in the all possible combination list
            for subsettingCombo in allPossibleSubsettingCombos:
                if subsettingCombo in rowList:
                    actualSubsettingCombos.append(subsettingCombo)
            #actualSubsettingCombos = allPossibleSubsettingCombos
        #Use these levels to cross section the fulldf, generating a list of subsetted dfs that will each have their own figure
        allPossibleSubsettedDfList = []
        for actualSubsettingCombo in actualSubsettingCombos:
            possibleSubsettedDf = fulldf.xs(actualSubsettingCombo, level=figureLevelNames)
            allPossibleSubsettedDfList.append(possibleSubsettedDf)
    else:
        actualSubsettingCombos = ['All']
        allPossibleSubsettedDfList = [fulldf]
    actualLevelValueDfList = []
    #Go through each subsetteddf, and only grab rows with level values that are selected
    for possibleSubsettedDf in allPossibleSubsettedDfList:
        allPossibleLevelValueCombos = itertools.product(*withinFigureSubsettedLevelValues)
        if exclusionLevel not in fulldf.index.names:
            rowList = []
            for row in range(possibleSubsettedDf.shape[0]):
                allCurrentLevelValues = possibleSubsettedDf.iloc[row,:].name
                rowList.append(allCurrentLevelValues)
            actualLevelValueCombos = []
            #From original dataframe; select all rows that appear in the all possible level value combination list
            levelValueRowList = []
            for levelValueCombo in allPossibleLevelValueCombos:
                if levelValueCombo in rowList:
                    indices = [i for i, x in enumerate(rowList) if x == levelValueCombo]
                    levelValueRowList+=indices
            actualLevelValueDf = possibleSubsettedDf.iloc[levelValueRowList,:]
            actualLevelValueDfList.append(actualLevelValueDf)
        else:
            unstackedDf = possibleSubsettedDf.unstack(exclusionLevel)
            allLevelValues = []
            for row in range(unstackedDf.shape[0]):
                allCurrentLevelValues = unstackedDf.iloc[row,:].name
                allLevelValues.append(allCurrentLevelValues)
            realLevelValueCombos = []
            for possibleLevelValueCombo in allPossibleLevelValueCombos:
                if possibleLevelValueCombo in allLevelValues:
                    realLevelValueCombos.append(possibleLevelValueCombo)
            subsettingList = list(map(list, zip(*realLevelValueCombos)))
            realSubsettingList = []
            for subset in subsettingList:
                newSubset = list(pd.unique(subset))
                realSubsettingList.append(newSubset)
            actualSubsettingList = []
            print(realSubsettingList)
            realsubset = 0
            for subset,levelName in enumerate(possibleSubsettedDf.index.names):
                print(levelName)
                print(subset)
                if levelName == exclusionLevel:
                    actualSubsettingList.append(slice(None))
                else:
                    actualSubsettingList.append(realSubsettingList[realsubset])
                    realsubset+=1
            actualLevelValueDf = possibleSubsettedDf.loc[tuple(actualSubsettingList),:]
            actualLevelValueDfList.append(actualLevelValueDf)
    if not isinstance(actualLevelValueDfList,list):
        actualLevelValueDfList = [actualLevelValueDfList]
    if not isinstance(actualSubsettingCombos,list):
        actualSubsettingCombos = ['All']

    #Remove columns from non postprocessed data
    if 'Dimension' not in fulldf.columns[0]:
        for i in range(len(actualLevelValueDfList)):
            actualLevelValueDfList[i].columns.name = ''
    return actualLevelValueDfList,actualSubsettingCombos,figureLevelNames,levelValuesPlottedIndividually

def createFacetPlotName(folderName,dataType,plotType,subPlotType,legendParameterToLevelNameDict,subsettedDfTitle,levelsPlottedIndividually,useModifiedDf,plotOptions):
    delimiter1 = '-'
    delimiter2 = ','

    legendParameterString = delimiter2.join(list(legendParameterToLevelNameDict.keys()))

    flattened_list = []
    print(legendParameterToLevelNameDict)
    for val in legendParameterToLevelNameDict.values():
        if isinstance(val, (list,)):
            print('wat')
            for val2 in val:
                flattened_list.append(val2)
        else:
            print('wat2')
            flattened_list.append(val)

    levelNameString = delimiter2.join(flattened_list)
    figureLevelNameString = delimiter2.join(list(map(str,subsettedDfTitle)))

    if len(levelsPlottedIndividually) == 0:
        individualLevelString = 'all'
    else:
        individualLevelString = delimiter2.join(list(map(str,levelsPlottedIndividually)))
    if useModifiedDf:
        modifiedString = '-modified'
    else:
        modifiedString = ''

    axisScalingStringList = []
    for axis in plotOptions:
        if 'X' in axis:
            if plotOptions[axis]['axisScaling'] == 'Logarithmic':
                axisScalingStringList.append('logX')
            elif plotOptions[axis]['axisScaling'] == 'Biexponential':
                axisScalingStringList.append('biexpX')
            else:
                axisScalingStringList.append('linX')
        else:
            if plotOptions[axis]['axisScaling'] == 'Logarithmic':
                axisScalingStringList.append('logY')
            elif plotOptions[axis]['axisScaling'] == 'Biexponential':
                axisScalingStringList.append('biexpY')
            else:
                axisScalingStringList.append('linY')
    axisScalingString = delimiter2.join(axisScalingStringList)

    if len(subsettedDfTitle) == 0 or subsettedDfTitle == 'All':
        initialString = delimiter1.join([subPlotType,dataType,folderName,legendParameterString,levelNameString,axisScalingString])
    else:
        initialString = delimiter1.join([subPlotType,dataType,folderName,legendParameterString,levelNameString,figureLevelNameString,axisScalingString])
    fullTitleString = initialString+modifiedString
    if '/' in fullTitleString:
        fullTitleString = fullTitleString.replace("/", "_")
    if '.' in fullTitleString:
        fullTitleString = fullTitleString.replace(".", "_")
    if ' ' in fullTitleString:
        fullTitleString = fullTitleString.replace(" ", "_")
    return fullTitleString

def subsetOriginalLevelValueOrders(unsubsettedOrders,subsettedDf,parameter):
    subsettedValues = list(pd.unique(subsettedDf.index.get_level_values(parameter)))
    subsettedOrders = []
    for unsubsettedValue in unsubsettedOrders:
        if unsubsettedValue in subsettedValues:
            subsettedOrders.append(unsubsettedValue)
    return subsettedOrders

def plotFacetedFigures(folderName,plotType,subPlotType,dataType,subsettedDfList,subsettedDfListTitles,figureParameters,levelsPlottedIndividually,useModifiedDf,fulldf,plotOptions,legendParameterToLevelNameDict,addDistributionPoints,alternateTitle='',originalLevelValueOrders = {},subfolderName='',context='notebook',height=3,aspect=1,titleBool='yes',colwrap=5,legendBool='yes',outlierBool='no',outlierZScore=3,plotAllVar=True,titleAdjust='',plotSpecificDict={}):
    sns.set_context(context)
    sns.set_palette(sns.color_palette())
    global hei,asp,titleVar,legendVar
    hei = height
    asp = aspect
    titleVar = titleBool
    legendVar = legendBool
    zScoreCutoff = 3
    #Has issues with repeated values (aka CD54 shows up in TCells and APCs)
    for subsettedDf,subsettedDfTitle in zip(subsettedDfList,subsettedDfListTitles):
        if outlierBool == 'yes':
            beforeOutlierRemoval = subsettedDf.shape[0]
            if plotOptions['Y']['axisScaling'] == 'Logarithmic':
                minVal = min(subsettedDf.values)
                if minVal <= 0:
                    newSubsettedDf = subsettedDf.iloc[:,:]+abs(minVal)+1
                else:
                    newSubsettedDf = subsettedDf.copy()
                newSubsettedDf = np.log10(newSubsettedDf)
                subsettedDf = subsettedDf[(np.abs(stats.zscore(newSubsettedDf)) < outlierZScore).all(axis=1)]
            else:
                subsettedDf = subsettedDf[(np.abs(stats.zscore(subsettedDf)) < outlierZScore).all(axis=1)]
            afterOutlierRemoval = subsettedDf.shape[0]
            print(str(beforeOutlierRemoval-afterOutlierRemoval)+' outliers removed!')

        #Assign all levels to plot parameters in catplot/relplot; reassign x/y axis level names to desired x/y axis titles
        kwargs = {}
        facetgridkwargs = {}
        for parameter in legendParameterToLevelNameDict:
            if parameter == 'Y Axis Values':
                if subPlotType == 'heatmap':
                    kwargs['y'] = legendParameterToLevelNameDict[parameter]
            else:
                currentLevel = legendParameterToLevelNameDict[parameter]
            print(originalLevelValueOrders)
            if parameter == 'Color':
                kwargs['hue'] = currentLevel
                if len(originalLevelValueOrders) != 0 and currentLevel in originalLevelValueOrders.keys():
                    kwargs['hue_order'] = subsetOriginalLevelValueOrders(originalLevelValueOrders[kwargs['hue']],subsettedDf,kwargs['hue'])
            elif parameter == 'Size':
                kwargs['size'] = currentLevel
                if len(originalLevelValueOrders) != 0 and currentLevel in originalLevelValueOrders.keys():
                    kwargs['size_order'] = subsetOriginalLevelValueOrders(originalLevelValueOrders[kwargs['size']],subsettedDf,kwargs['size'])
            elif parameter == 'Marker':
                kwargs['style'] = currentLevel
            elif parameter == 'Order':
                subsettedDf.index.set_names(plotOptions['X']['axisTitle'],level=currentLevel,inplace=True)
                kwargs['x'] = plotOptions['X']['axisTitle']
                if len(originalLevelValueOrders) != 0 and currentLevel in originalLevelValueOrders.keys():
                    kwargs['order'] = subsetOriginalLevelValueOrders(originalLevelValueOrders[kwargs['x']],subsettedDf,kwargs['x'])
            elif parameter == 'Column':
                kwargs['col'] = currentLevel
                unorderedLevelValues = list(pd.unique(subsettedDf.index.get_level_values(currentLevel)))
                if 'Concentration' in currentLevel and 'M' in unorderedLevelValues[0]:
                    levelValues = sortSINumerically(unorderedLevelValues,True,True)[0]
                    kwargs['col_order'] = levelValues
                else:
                    if len(originalLevelValueOrders) != 0 and currentLevel in originalLevelValueOrders.keys():
                        kwargs['col_order'] = subsetOriginalLevelValueOrders(originalLevelValueOrders[kwargs['col']],subsettedDf,kwargs['col'])
                    else:
                        kwargs['col_order'] = unorderedLevelValues
                facetgridkwargs['col'] = kwargs['col']
                facetgridkwargs['col_order'] = kwargs['col_order']
            elif parameter == 'Row':
                kwargs['row'] = currentLevel
                unorderedLevelValues = list(pd.unique(subsettedDf.index.get_level_values(currentLevel)))
                if 'Concentration' in currentLevel and 'M' in unorderedLevelValues[0]:
                    levelValues = sortSINumerically(unorderedLevelValues,True,True)[0]
                    kwargs['row_order'] = levelValues
                else:
                    levelValues = unorderedLevelValues
                    if len(originalLevelValueOrders) != 0 and currentLevel in originalLevelValueOrders.keys():
                        kwargs['row_order'] = subsetOriginalLevelValueOrders(originalLevelValueOrders[kwargs['row']],subsettedDf,kwargs['row'])
                    else:
                        kwargs['row_order'] = unorderedLevelValues
                facetgridkwargs['row'] = kwargs['row']
                facetgridkwargs['row_order'] = kwargs['row_order']
            elif parameter == 'X Axis Values':
                if len(plotOptions['X']['axisTitle']) > 1 and dataType != 'dr':
                    subsettedDf.index.set_names(plotOptions['X']['axisTitle'],level=currentLevel,inplace=True)
                    kwargs['x'] = plotOptions['X']['axisTitle']
                else:
                    kwargs['x'] = currentLevel
            else:
                pass
        #Assign y axis (or color bar axis) parameter
        #if subPlotType not in ['kde','histogram']:
        if dataType == 'dr':
            kwargs['x'] = plotOptions['X']['axisTitle']
            kwargs['y'] = plotOptions['Y']['axisTitle']
        else:
            if subPlotType != 'heatmap':
                if 'Statistic' in figureParameters:
                    subsettedDf.columns = [subsettedDfTitle[figureParameters.index('Statistic')]]
                    kwargs['y'] = subsettedDfTitle[figureParameters.index('Statistic')]
                elif 'Node' in subsettedDf.columns[0]:
                    kwargs['x'] = subsettedDf.columns[0]
                    kwargs['y'] = subsettedDf.columns[1]
                else:
                    subsettedDf.columns = [plotOptions['Y']['axisTitle']]
                    kwargs['y'] = plotOptions['Y']['axisTitle']
            else:
                if 'Statistic' in figureParameters:
                    subsettedDf.columns = [subsettedDfTitle[figureParameters.index('Statistic')]]
                    kwargs['z'] = subsettedDfTitle[figureParameters.index('Statistic')]
                else:
                    subsettedDf.columns = [plotOptions['Colorbar']['axisTitle']]
                    kwargs['z'] = plotOptions['Colorbar']['axisTitle']
        #else:
        #pass

        if dataType == 'singlecell':
            plottingDf = subsettedDf.copy()
            #plottingDf = subsettedDf.stack().to_frame('GFI')
        else:
            plottingDf = subsettedDf.copy()
        #Converts wide form dataframe into long form required for cat/relplot
        plottingDf = plottingDf.reset_index()
        print(plottingDf)
        #Use plot options file to initialize numeric x axis ordering
        #NEED TO GET WORKING WITH HEATMAPS
        if subPlotType != 'heatmap':
            if plotType != '1d':
                if plotOptions['X']['numeric']:
                    currentLevelValues = list(plottingDf[kwargs['x']])
                    sortedOldLevelValues,newLevelValues = sortSINumerically(currentLevelValues,False,True)
                    #Need to interpret parenthetical units for x to get 1e9
                    if 'M)' in kwargs['x']:
                        s = kwargs['x']
                        units = '1'+s[s.find("(")+1:s.find(")")]
                        scaledSortedUnits,sortedUnits = sortSINumerically([units],False,True)
                    else:
                        sortedUnits = [1]
                    scaledNewLevelValues = [float(i) / float(sortedUnits[0]) for i in newLevelValues]
                    plottingDf[kwargs['x']] = scaledNewLevelValues
                    if plotType == 'categorical':
                        kwargs['order'] = scaledNewLevelValues
        else:
            pass
        col_wrap_min = colwrap
        #Make sure there are not more than 6 plots in a row (if no row facet variable)
        if 'row' not in facetgridkwargs.keys():
            if 'col' in facetgridkwargs.keys():
                colwrap = min([len(kwargs['col_order']),col_wrap_min])
                kwargs['col_wrap'] = colwrap

        #HACK; NEED TO THINK OF BETTER WAY TO DO THIS:
        filePlottingFolderName = pickle.load(open(path+'scripts/gui/plotting/plottingFolderName.pkl','rb'))
        global plotFolderName
        plotFolderName = path+'figures/'+filePlottingFolderName

        if plotAllVar:
            fullTitleString = createFacetPlotName(folderName,dataType,plotType,subPlotType,legendParameterToLevelNameDict,subsettedDfTitle,levelsPlottedIndividually,useModifiedDf,plotOptions)
            if 'temporaryFirstPlot.png' in os.listdir(plotFolderName):
                subprocess.run(['rm',plotFolderName+'/temporaryFirstPlot.png'])
        else:
            fullTitleString = 'temporaryFirstPlot'
        if len(subsettedDf.index) > 0:
            plotSubsettedFigure(subsettedDf,plottingDf,kwargs,facetgridkwargs,plotSpecificDict,plotType,subPlotType,dataType,fullTitleString,plotOptions,subsettedDfTitle,addDistributionPoints,alternateTitle,subfolderName,titleAdjust)
        if not plotAllVar:
            break
    sns.set_context('notebook')

def reorderKwargs(oldKwargs,kwargs):
    reorderedKwargs = {}
    for key in kwargs:
        if key in oldKwargs:
            reorderedList = []
            if isinstance(oldKwargs[key],list) or isinstance(oldKwargs[key],tuple):
                for val in oldKwargs[key]:
                    for val2 in kwargs[key]:
                        if isinstance(val,str):
                            if val == val2.strip():
                                reorderedList.append(val2)
                                break
                        else:
                            print('val not in val2')
                if len(reorderedList) == 0:
                    reorderedKwargs[key] = kwargs[key]
                else:
                    reorderedKwargs[key] = reorderedList
            else:
                reorderedKwargs[key] = kwargs[key]
        else:
            reorderedKwargs[key] = kwargs[key]
    return reorderedKwargs

#Will not be needed when seaborn 0.9.1 releases
def sanitizeSameValueLevels(plottingDf,kwargs):
    oldKwargs = kwargs.copy()
    unsanitizedPlottingDf = plottingDf.copy()
    #First find levels that have values that are the same as other levels:
    #Get numeric axes levels
    numericAxes = []
    for axis in ['x','y','z']:
        if axis in kwargs.keys():
            numericAxes.append(kwargs[axis])
    #Exclude levels used for numeric axes (x,y,color)
    nonNumericNameList = []
    for levelName in plottingDf.columns:
        if levelName not in numericAxes:
            nonNumericNameList.append(levelName)
    #Get all possible pairs of nonnumericlevels
    possiblePairs = list(itertools.combinations(nonNumericNameList,2))
    overlappingPairs = []
    for possiblePair in possiblePairs:
        uniqueLevelValues1 = pd.unique(plottingDf[possiblePair[0]])
        uniqueLevelValues2 = pd.unique(plottingDf[possiblePair[1]])
        #If overlap, add both levelnames to list
        if bool(set(uniqueLevelValues1) & set(uniqueLevelValues2)):
            overlappingPairs.append(possiblePair)
    #Add dummy spaces to make level values different while not affecting the legend
    #NEED TO THINK ABOUT HOW THIS AFFECTS NUMERIC VARIABLES
    #NEED TO THINK ABOUT HOW TO DEAL WITH VARIABLES I NEED TO SORT NUMERICALLY; MOST LIKELY BY DETERMINING ORDER BEFORE SANITIZING AND ADDING SPACES TO ORDER
    for overlappingPair in overlappingPairs:
        newVals = []
        for value in plottingDf[overlappingPair[1]]:
            newval = str(value) + ' '
            newVals.append(newval)
        plottingDf[overlappingPair[1]] = newVals
        for kwarg,kwargParent in zip(['row_order','col_order','hue_order','size_order'],['row','col','hue','size']):
            if kwarg in kwargs.keys():
                if kwargs[kwargParent] == overlappingPair[1]:
                    kwargs[kwarg] = list(pd.unique(plottingDf[overlappingPair[1]]))
    kwargs = reorderKwargs(oldKwargs,kwargs)
    return plottingDf,kwargs

def plotSubsettedFigure(subsettedDf,plottingDf,kwargs,facetgridkwargs,plotSpecificKwargs,plotType,subPlotType,dataType,fullTitleString,plotOptions,subsettedDfTitle,addDistributionPoints,alternateTitle,subfolderName,titleAdjust):

    titleBool = True
    secondPathBool = False

    #Sanitize plottingDf to have every level value be unique across levels
    if subPlotType != 'heatmap':
        plottingDf,kwargs = sanitizeSameValueLevels(plottingDf,kwargs)

    #Add in sharex/sharey options
    if plotType != '1d':
        print(plotOptions['X']['share'])
        print(plotOptions['Y']['share'])
        facetKwargs = {'sharex':plotOptions['X']['share'],'sharey':plotOptions['Y']['share']}
    else:
        facetKwargs = {'sharex':False,'sharey':plotOptions['Y']['share']}

    if alternateTitle !=  '':
        fullTitleString = alternateTitle

    auxillaryKwargs = {}
    auxillaryKwargs['plotType'] = plotType
    auxillaryKwargs['subPlotType'] = subPlotType
    auxillaryKwargs['facetgridkwargs'] = facetgridkwargs
    auxillaryKwargs['plotspecifickwargs'] = plotSpecificKwargs
    #if hei != 5 and asp != 1:
    if plotType == '1d':
        plotOptions['Y']['figureDimensions'] = {'height':hei,'aspect':asp}
    else:
        plotOptions['X']['figureDimensions'] = {'height':hei,'aspect':asp}
    #else:
    #    plotOptions['X']['figureDimensions'] = {}
    #Use appropriate facet plot
    #1D plots: (Histograms and KDEs); use facetgrid with appropriate axis level seaborn function
    if plotType == '1d':
        auxillaryKwargs['dataType'] = dataType
        facetPlotType = fp1D
        #Fix duck typing issue with replots: https://github.com/mwaskom/seaborn/issues/1653
        if 'hue' in kwargs and isinstance(plottingDf[kwargs['hue']][0],str):
            if plottingDf[kwargs['hue']][0].isnumeric():
                for i,x in enumerate(kwargs['hue_order']):
                    kwargs['hue_order'][i] = "$%s$" % x
                plottingDf[kwargs['hue']] = ["$%s$" % x for x in plottingDf[kwargs['hue']]]
    #1.5D/categorical plots (bar/point/box etc.); use catplot figure level seaborn function
    elif plotType == 'categorical':
        auxillaryKwargs['addDistributionPoints'] = addDistributionPoints
        facetPlotType = fpCategorical
    #2D plots (line and scatter); use relplot figure level seaborn function
    elif plotType == '2d':
        facetPlotType = fp2D
        #Fix duck typing issue with replots: https://github.com/mwaskom/seaborn/issues/1653
        if 'hue' in kwargs and isinstance(plottingDf[kwargs['hue']][0],str):
            if plottingDf[kwargs['hue']][0].isnumeric():
                for i,x in enumerate(kwargs['hue_order']):
                    kwargs['hue_order'][i] = "$%s$" % x
                plottingDf[kwargs['hue']] = ["$%s$" % x for x in plottingDf[kwargs['hue']]]
        if 'size' in kwargs and isinstance(plottingDf[kwargs['size']][0],str):
            if plottingDf[kwargs['size']][0].isnumeric():
                plottingDf[kwargs['size']] = ["$%s$" % x for x in plottingDf[kwargs['size']]]
    #3D plots (heatmaps and 3d scatter/lineplots); use facet grid with appropriate axis level seaborn function
    elif plotType == '3d':
        facetPlotType = fp3D
    fg = facetPlotType.plot(plottingDf,subsettedDf,kwargs,facetKwargs,auxillaryKwargs,plotOptions)
    #SupTitle
    if titleVar == 'yes':
        #Make room for suptitle
        if titleAdjust == '':
            if 'row' in kwargs:
                if len(kwargs['row_order']) <=2:
                    plt.subplots_adjust(top=0.9)
                else:
                    plt.subplots_adjust(top=0.9+len(kwargs['row_order'])*0.005)
            else:
                plt.subplots_adjust(top=0.8)
        else:
            plt.subplots_adjust(top=float(titleAdjust))
        if subsettedDfTitle != 'All':
            subsettedDfTitle = list(map(str,subsettedDfTitle))
        else:
            subsettedDfTitle = [subsettedDfTitle]
        #Do not include placeholder celltypes in suptitle
        if 'NotApplicable' in subsettedDfTitle:
            subsettedDfTitle.remove('NotApplicable')
        plt.suptitle('-'.join(subsettedDfTitle),fontsize = 'x-large',fontweight='bold')

    #Legend
    if legendVar == 'no':
        fg._legend.remove()

    #Save figure
    if subfolderName != '':
        if subfolderName not in os.listdir(plotFolderName):
            subprocess.run(['mkdir',plotFolderName+'/'+subfolderName])
        fg.fig.savefig(plotFolderName+'/'+subfolderName+'/'+fullTitleString+'.png',bbox_inches='tight')
    else:
        fg.fig.savefig(plotFolderName+'/'+fullTitleString+'.png',bbox_inches='tight')
    if secondPathBool:
        fg.fig.savefig('../../outputFigures/'+fullTitleString+'.png',bbox_inches='tight')
    print(fullTitleString+' plot saved')
    plt.clf()
