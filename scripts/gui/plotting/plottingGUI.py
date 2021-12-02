#!/usr/bin/env python3
import pickle,sys,os,json,math,subprocess,string
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import tkinter as tk
import tkinter.ttk
import pandas as pd
import scripts.gui.plotting.facetPlotLibrary as fpl
import scripts.gui.plotting.interactiveGUIElements as ipe
from scripts.process.adapt_dataframes import set_standard_order

expParamDict = {'cyt':'cyt','cell':'cell','prolif':'cell','singlecell':'cell'}

#Get level names and values into an easily accessible dictionary
def createLabelDict(df,discretized_time=False,sortedValues={}):
    #fulldf = df.stack()
    fulldf = df.copy()
    labelDict = {}
    for i in range(fulldf.index.nlevels):
        levelName = fulldf.index.levels[i].name
        parameterList = ['Event','event']
        if not discretized_time:
            parameterList += ['Time']
        if levelName not in parameterList:
            labelDict[levelName] = list(pd.unique(fulldf.index.get_level_values(levelName)))
    if len(sortedValues) != 0:
        for level in sortedValues:
            labelDict[level] = sortedValues[level]
    return labelDict

class checkUncheckAllButton(tk.Button):
    def __init__(self,parent,checkButtonList,**kwargs):
        tk.Button.__init__(self,parent,**kwargs)
        self.checkButtonList = checkButtonList
        self.parent = parent

    def checkAll(self):
        for checkButton in self.checkButtonList:
            checkButton.select()

    def uncheckAll(self):
        for checkButton in self.checkButtonList:
            checkButton.deselect()

#FOR PLOT DATA SELECTION
class selectLevelsPage(tk.Frame):
    def __init__(self, master,inputdf,hpage):
        tk.Frame.__init__(self, master)
        labelWindow = tk.Frame(self)
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        global homePage
        global experimentDf
        homePage = hpage
        experimentDf = inputdf
        global trueLabelDict
        trueLabelDict = {}
        sortedValues = set_standard_order(experimentDf.copy().reset_index(),returnSortedLevelValues=True)
        trueLabelDict = createLabelDict(experimentDf.copy(),sortedValues=sortedValues)
        #trueLabelDict = createLabelDict(experimentDf)

        global plotType
        global subPlotType
        global addDistributionPoints
        addDistributionPoints = False
        global multipleDimensions
        if len(inputdf.columns) > 2:
            multipleDimensions = True
            if 'Node 1' in inputdf.columns:
                plotType,subPlotType = '2d/line'.split('/')
            else:
                plotType,subPlotType = '2d/scatter'.split('/')
        else:
            multipleDimensions = False
            plotType,subPlotType = '2d/line'.split('/')

        global figureLevelList,fullFigureLevelBooleanList
        fullFigureLevelBooleanList = []
        figureLevelList = []

        # TODO: there is a bug when some level is not selected, as if the
        # DataFrame passed to the plotting function were ending up empty
        # (frbourassa, 2020-aug-18)
        l1 = tk.Label(labelWindow, text="""Which levels names do you want to be included within this figure??:""").pack()
        mainWindow = tk.Frame(self)
        levelNameCheckButtons = []
        checkButtonVariableList = []
        for levelName,i in zip(trueLabelDict.keys(),range(len(trueLabelDict.keys()))):
            includeLevelBool = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(mainWindow, text=levelName,padx = 20, variable=includeLevelBool,onvalue=True)
            cb.grid(row=i+3,column=1,sticky=tk.W)
            cb.select()
            levelNameCheckButtons.append(cb)
            checkButtonVariableList.append(includeLevelBool)

        checkButtonWindow = tk.Frame(self)
        checkAllButton1 = checkUncheckAllButton(checkButtonWindow,levelNameCheckButtons, text='Check All')
        checkAllButton1.configure(command=checkAllButton1.checkAll)
        checkAllButton1.pack(side=tk.LEFT)

        uncheckAllButton1 = checkUncheckAllButton(checkButtonWindow,levelNameCheckButtons, text='Uncheck All')
        uncheckAllButton1.configure(command=checkAllButton1.uncheckAll)
        uncheckAllButton1.pack(side=tk.LEFT)

        checkButtonWindow.pack(side=tk.TOP)
        mainWindow.pack(side=tk.TOP,padx=10)

        def collectInputs():
            includeLevelList = []
            for checkButtonVariable in checkButtonVariableList:
                includeLevelList.append(checkButtonVariable.get())
            for figureLevelBool,levelName in zip(includeLevelList,trueLabelDict):
                if figureLevelBool:
                    figureLevelList.append(levelName)
                    fullFigureLevelBooleanList.append(True)
                else:
                    fullFigureLevelBooleanList.append(False)
            master.switch_frame(selectLevelValuesPage,assignLevelsToParametersPage,trueLabelDict)

        def quitCommand():
            exitBoolean = True
            quit()

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        # TODO: bug? all processed input files are read again when pressing "Back"
        # (frbourassa, 2020-aug-18)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(homePage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quitCommand()).pack(in_=buttonWindow,side=tk.LEFT)

class selectLevelValuesPage(tk.Frame):
    #(selectLevelValuesPage,SelectDimensionsPage,trueLabelDict,DataSelectionHomePage,finalSwitchPage,folderName,secondaryhomepage,processType)
    #master.switch_frame(selectLevelValuesPage,SelectDimensionsPage,trueLabelDict,DataSelectionHomePage,backpage,finalSwitchPage,folderName,secondaryhomepage,processType)
    def __init__(self, master,switchPage,trueLabelDict):
        tk.Frame.__init__(self, master)

        includeLevelValueList = []

        labelWindow = tk.Frame(self)
        labelWindow.pack(side=tk.TOP,padx=10,fill=tk.X,expand=True)

        l1 = tk.Label(labelWindow, text='Which specific level values do you want to include in the figure?',pady=10).grid(row=0,column = 0,columnspan=len(trueLabelDict)*6)
        levelValueCheckButtonList = []
        overallCheckButtonVariableList = []
        checkAllButtonList = []
        uncheckAllButtonList = []
        i=0
        maxNumLevelValues = 0
        for levelName in trueLabelDict:
            j=0
            levelCheckButtonList = []
            levelCheckButtonVariableList = []
            levelLabel = tk.Label(labelWindow, text=levelName+':')
            levelLabel.grid(row=1,column = i*6,sticky=tk.N,columnspan=5)
            for levelValue in trueLabelDict[levelName]:
                includeLevelValueBool = tk.BooleanVar()
                cb = tk.Checkbutton(labelWindow, text=levelValue, variable=includeLevelValueBool)
                cb.grid(row=j+4,column=i*6+2,columnspan=2,sticky=tk.W)
                labelWindow.grid_columnconfigure(i*6+3,weight=1)
                cb.select()
                levelCheckButtonList.append(cb)
                levelCheckButtonVariableList.append(includeLevelValueBool)
                j+=1

            checkAllButton1 = checkUncheckAllButton(labelWindow,levelCheckButtonList, text='Check All')
            checkAllButton1.configure(command=checkAllButton1.checkAll)
            checkAllButton1.grid(row=2,column=i*6,sticky=tk.N,columnspan=3)
            checkAllButtonList.append(checkAllButton1)

            uncheckAllButton1 = checkUncheckAllButton(labelWindow,levelCheckButtonList, text='Uncheck All')
            uncheckAllButton1.configure(command=checkAllButton1.uncheckAll)
            uncheckAllButton1.grid(row=2,column=i*6+3,sticky=tk.N,columnspan=3)
            uncheckAllButtonList.append(checkAllButton1)

            levelValueCheckButtonList.append(levelCheckButtonList)
            overallCheckButtonVariableList.append(levelCheckButtonVariableList)
            if len(trueLabelDict[levelName]) > maxNumLevelValues:
                maxNumLevelValues = len(trueLabelDict[levelName])
            i+=1

        def collectInputs():
            for checkButtonVariableList in overallCheckButtonVariableList:
                tempLevelValueList = []
                for checkButtonVariable in checkButtonVariableList:
                    tempLevelValueList.append(checkButtonVariable.get())
                includeLevelValueList.append(tempLevelValueList)
            #master.switch_frame(assignLevelsToParametersPage)
            master.switch_frame(switchPage,includeLevelValueList)

        def quitCommand():
            exitBoolean = True
            quit()

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)

        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).grid(row=maxNumLevelValues+4,column=0)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(selectLevelsPage,experimentDf,homePage)).grid(row=maxNumLevelValues+4,column=1)
        tk.Button(buttonWindow, text="Quit",command=lambda: quitCommand()).grid(row=maxNumLevelValues+4,column=2)

class assignLevelsToParametersPage(tk.Frame):

    def __init__(self, master,temp):
        parameterTypeDict = {
                'categorical':['Color','Order', 'Row', 'Column','None'],
                '1d':['Color','Row','Column','None'],
                '3d':['Row','Column','X Axis Values','Y Axis Values']}

        if not multipleDimensions:
            parameterTypeDict['2d'] = ['Marker','Color','Size','Row','Column','X Axis Values','None']
        else:
            parameterTypeDict['2d'] = ['Marker','Color','Size','Row','Column','None']

        tk.Frame.__init__(self, master)
        global includeLevelValueList
        includeLevelValueList = temp
        global parametersSelected
        parametersSelected = {}
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=tk.TOP,padx=10,pady=10)

        tk.Label(mainWindow, text='Which plotting parameter do you want to assign to each of your figure levels?',pady=10).grid(row=0,column = 0,columnspan = len(figureLevelList))
        rblist = []
        parameterVarList = []
        for figureLevel,figureLevelIndex in zip(figureLevelList,range(len(figureLevelList))):
            v = tk.IntVar()
            temprblist = []
            levelLabel = tk.Label(mainWindow, text=figureLevel+':')
            levelLabel.grid(row=1,column=figureLevelIndex,sticky=tk.NW)
            for plottingParameter,parameterIndex in zip(parameterTypeDict[plotType],range(len(parameterTypeDict[plotType]))):
                rb = tk.Radiobutton(mainWindow, text=plottingParameter,padx = 20, variable=v, value=parameterIndex)
                rb.grid(row=parameterIndex+2,column=figureLevelIndex,sticky=tk.NW)
                temprblist.append(rb)
            rblist.append(temprblist)
            parameterVarList.append(v)
        if multipleDimensions:
            mainWindow2 = tk.Frame(self)
            mainWindow2.pack(side=tk.TOP,padx=10,pady=10)
            axes = ['X Axis Values','Y Axis Values']
            tk.Label(mainWindow2, text='Which dimension do you want to assign to each of your figure axes?',pady=10).grid(row=0,column = 0,columnspan = len(axes))
            rblist2 = []
            parameterVarList2 = []
            for i,axis in enumerate(axes):
                v2 = tk.IntVar()
                temprblist2 = []
                levelLabel2 = tk.Label(mainWindow2, text=axis+':')
                levelLabel2.grid(row=1,column=i,sticky=tk.NW)
                for j,dimension in enumerate(experimentDf.columns):
                    rb = tk.Radiobutton(mainWindow2, text=dimension,padx = 20, variable=v2, value=j)
                    rb.grid(row=j+2,column=i,sticky=tk.NW)
                    temprblist2.append(rb)
                rblist2.append(temprblist2)
                parameterVarList2.append(v2)

        def collectInputs():
            for parameterVar,levelName in zip(parameterVarList,figureLevelList):
                if parameterTypeDict[plotType][parameterVar.get()] not in parametersSelected.keys():
                    parametersSelected[parameterTypeDict[plotType][parameterVar.get()]] = levelName
                else:
                    if not isinstance(parametersSelected[parameterTypeDict[plotType][parameterVar.get()]],list):
                        parametersSelected[parameterTypeDict[plotType][parameterVar.get()]] = [parametersSelected[parameterTypeDict[plotType][parameterVar.get()]]]+[levelName]
                    else:
                        parametersSelected[parameterTypeDict[plotType][parameterVar.get()]].append(levelName)
            if multipleDimensions:
                for i,axis in enumerate(axes):
                    parametersSelected[axis] = list(experimentDf.columns)[parameterVarList2[i].get()]
                print(parametersSelected)
            else:
                if plotType == '2d' and 'X Axis Values' not in parametersSelected:
                    if 'Feature' in experimentDf.index.names and 'Node' not in experimentDf.columns[0]:
                        parametersSelected['X Axis Values'] = 'Time'
                    else:
                        parametersSelected['X Axis Values'] = experimentDf.columns[0]
            master.switch_frame(plotElementsGUIPage)

        def quitCommand():
            exitBoolean = True
            quit()

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)

        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).grid(row=len(parameterTypeDict[plotType])+2,column=0)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(selectLevelValuesPage,assignLevelsToParametersPage,trueLabelDict)).grid(row=len(parameterTypeDict[plotType])+2,column=1)
        tk.Button(buttonWindow, text="Quit",command=lambda: quitCommand()).grid(row=len(parameterTypeDict[plotType])+2,column=2)

def getDefaultAxisTitles():
        xaxistitle = ''
        yaxistitle = ''
        cbartitle = ''
        cytokineDefault = 'Concentration (nM)'
        if not multipleDimensions:
            if plotType == '1d':
                xaxistitle = 'MFI'
                yaxistitle = 'Frequency'
            else:
                if plotType == 'categorical':
                    xaxistitle = parametersSelected['Order']
                    if dataType == 'cyt':
                        yaxistitle = cytokineDefault
                else:
                    xaxistitle = parametersSelected['X Axis Values']
                    if 'Node' in experimentDf.columns[0]:
                        if plotType == '2d':
                            yaxistitle = experimentDf.columns[1]
                        else:
                            cbartitle = experimentDf.columns[1]
                    else:
                        if 'Feature' in experimentDf.index.names:
                            if plotType == '2d':
                                yaxistitle = 'value'
                            else:
                                cbartitle = 'value'
                        else:
                            if dataType == 'cyt':
                                if plotType == '2d':
                                    yaxistitle = cytokineDefault
                                else:
                                    cbartitle = cytokineDefault
            if xaxistitle == 'Time' and yaxistitle != 'value':
                xaxistitle += ' (hours)'
        else:
            xaxistitle = parametersSelected['X Axis Values']
            yaxistitle = parametersSelected['Y Axis Values']
        return [xaxistitle,yaxistitle,cbartitle]

class plotElementsGUIPage(tk.Frame):
    def __init__(self, master):
        tempDict = {}
        stackedDf = experimentDf.stack()
        for level in stackedDf.index.names:
            levelValues = list(pd.unique(stackedDf.index.get_level_values(level)))
            tempDict[level] = levelValues
        experimentParameters = {}
        experimentParameters['allLevelValues'] = tempDict
        """
        global dataType
        if 'cell' in pickleFileName:
            dataType = 'cell'
        elif 'cyt' in pickleFileName:
            dataType = 'cyt'
        elif 'prolif' in pickleFileName:
            dataType = 'prolif'
        else:
            dataType = ''
        """
        print('TLD 1')
        self.tld = trueLabelDict

        axisDict = {'categorical':['X','Y'],'1d':['Y'],'2d':['X','Y'],'3d':['X','Y','Colorbar']}
        scalingList = ['Linear','Logarithmic','Biexponential']
        axisSharingList = ['col','row','']
        axisTitleDefaults = getDefaultAxisTitles()

        tk.Frame.__init__(self, master)

        mainWindow = tk.Frame(self)
        mainWindow.pack(side=tk.TOP,padx=10,pady=10)

        tk.Label(mainWindow, text='Title: ').grid(row=1,column=0,sticky=tk.W)
        for scaling,scalingIndex in zip(scalingList,range(len(scalingList))):
            tk.Label(mainWindow, text=scaling+' Scaling: ').grid(row=scalingIndex+2,column=0,sticky=tk.W)
        tk.Label(mainWindow, text='Linear Range (Biexponential Scaling): ').grid(row=len(scalingList)+2,column=0,sticky=tk.W)
        tk.Label(mainWindow, text='Convert to numeric: ').grid(row=len(scalingList)+3,column=0,sticky=tk.W)
        tk.Label(mainWindow, text='Share axis across: ').grid(row=len(scalingList)+4,column=0,sticky=tk.W)
        tk.Label(mainWindow, text='Axis limits: ').grid(row=len(scalingList)+5,column=0,sticky=tk.W)

        entryList = []
        scalingVariableList = []
        radioButtonList = []
        checkButtonList = []
        checkButtonVarList = []
        radioButtonList2 = []
        radioButtonVarList2 = []
        linearRangeScalingList = []
        limitEntryList = []
        for axis,axisIndex in zip(axisDict[plotType],range(len(axisDict[plotType]))):
            tk.Label(mainWindow, text=axis+ ' Axis').grid(row=0,column=axisIndex+1)

            e1 = tk.Entry(mainWindow)
            e1.grid(row=1,column=axisIndex+1)
            e1.insert(0, axisTitleDefaults[axisIndex])
            entryList.append(e1)

            axisRadioButtonList = []
            v = tk.StringVar(value='Linear')
            for scaling,scalingIndex in zip(scalingList,range(len(scalingList))):
                rb = tk.Radiobutton(mainWindow,variable=v,value=scaling)
                rb.grid(row=scalingIndex+2,column=axisIndex+1)
                axisRadioButtonList.append(rb)
            radioButtonList.append(axisRadioButtonList)
            scalingVariableList.append(v)

            e2 = tk.Entry(mainWindow)
            e2.grid(row=len(scalingList)+2,column=axisIndex+1)
            linearRangeScalingList.append(e2)

            b = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(mainWindow,variable=b)
            cb.grid(row=len(scalingList)+3,column=axisIndex+1)
            checkButtonList.append(cb)
            checkButtonVarList.append(b)

            shareWindow = tk.Frame(mainWindow)
            shareWindow.grid(row=len(scalingList)+4,column=axisIndex+1)
            shareString = tk.StringVar(value='None')
            rb2a = tk.Radiobutton(shareWindow,variable=shareString,text='All',value='all')
            rb2b = tk.Radiobutton(shareWindow,variable=shareString,text='Row',value='row')
            rb2c = tk.Radiobutton(shareWindow,variable=shareString,text='Col',value='col')
            rb2d = tk.Radiobutton(shareWindow,variable=shareString,text='None',value='none')
            shareString.set('all')
            rb2a.grid(row=0,column=1)
            rb2b.grid(row=0,column=2)
            rb2c.grid(row=0,column=3)
            rb2d.grid(row=0,column=4)
            radioButtonList2.append([rb2a,rb2b])
            radioButtonVarList2.append(shareString)

            limitWindow = tk.Frame(mainWindow)
            limitWindow.grid(row=len(scalingList)+5,column=axisIndex+1)
            #ll = tk.Label(limitWindow,text='Lower:').grid(row=0,column=0)
            e3 = tk.Entry(limitWindow,width=5)
            e3.grid(row=0,column=1)
            #ul = tk.Label(limitWindow,text='Upper:').grid(row=0,column=2)
            e4 = tk.Entry(limitWindow,width=5)
            e4.grid(row=0,column=3)
            limitEntryList.append([e3,e4])

        def collectInputs(plotAllVar):
            plotOptions = {}
            for axis,axisIndex in zip(axisDict[plotType],range(len(axisDict[plotType]))):
                share = radioButtonVarList2[axisIndex].get()
                if share == 'none':
                    share = False
                plotOptions[axis] = {'axisTitle':entryList[axisIndex].get(),
                        'axisScaling':scalingVariableList[axisIndex].get(),
                        'linThreshold':linearRangeScalingList[axisIndex].get(),
                        'numeric':checkButtonVarList[axisIndex].get(),
                        'share':share,
                        'limit':[limitEntryList[axisIndex][0].get(),limitEntryList[axisIndex][1].get()]}

            plotSpecificDict = {}
            if subPlotType == 'kde':
                scaleBool = ipe.getRadiobuttonValues(modeScaleRadiobuttonVarsDict)['scale to mode']
                if scaleBool == 'yes':
                    plotSpecificDict['scaleToMode'] = True
                else:
                    plotSpecificDict['scaleToMode'] = False
                plotSpecificDict['smoothing'] = int(ipe.getSliderValues(smoothingSliderList,['smoothing'])['smoothing'])

            useModifiedDf = False
            sName = titleEntry.get()
            subsettedDfList,subsettedDfListTitles,figureLevels,levelValuesPlottedIndividually = fpl.produceSubsettedDataFrames(experimentDf,fullFigureLevelBooleanList,includeLevelValueList,self.tld)
            folderName = 'test'
            dataType = 'dr'
            fpl.plotFacetedFigures(folderName,plotType,subPlotType,dataType,subsettedDfList,subsettedDfListTitles,figureLevels,levelValuesPlottedIndividually,useModifiedDf,experimentDf,plotOptions,parametersSelected,addDistributionPoints,originalLevelValueOrders=experimentParameters['allLevelValues'],subfolderName=sName,context=ipe.getRadiobuttonValues(contextRadiobuttonVarsDict)['context'],height=float(heightEntry.get()),aspect=float(widthEntry.get()),titleBool=ipe.getRadiobuttonValues(plotTitleRadiobuttonVarsDict)['plotTitle'],colwrap=int(colWrapEntry.get()),legendBool=ipe.getRadiobuttonValues(legendRadiobuttonVarsDict)['legend'],outlierBool=ipe.getRadiobuttonValues(outlierRadiobuttonVarsDict)['remove outliers'],plotAllVar=plotAllVar,titleAdjust=titleAdjustEntry.get(),plotSpecificDict = plotSpecificDict)

        titleWindow = tk.Frame(self)
        titleWindow.pack(side=tk.TOP,pady=10)
        tk.Label(titleWindow,text='Enter subfolder for these plots (optional):').grid(row=0,column=0)
        titleEntry = tk.Entry(titleWindow,width=15)
        titleEntry.grid(row=0,column=1)

        miscOptionsWindow = tk.Frame(self)
        miscOptionsWindow.pack(side=tk.TOP,pady=10)

        contextWindow = tk.Frame(miscOptionsWindow)
        contextWindow.grid(row=0,column=0,sticky=tk.N)
        contextRadiobuttonList,contextRadiobuttonVarsDict = ipe.createParameterSelectionRadiobuttons(contextWindow,['context'],{'context':['notebook','talk','poster']})

        figureDimensionWindow = tk.Frame(miscOptionsWindow)
        figureDimensionWindow.grid(row=0,column=1,sticky=tk.N)
        tk.Label(figureDimensionWindow,text='figure dimensions').grid(row=0,column=0)
        tk.Label(figureDimensionWindow,text='height:').grid(row=1,column=0)
        tk.Label(figureDimensionWindow,text='width:').grid(row=2,column=0)
        heightEntry = tk.Entry(figureDimensionWindow,width=3)
        if plotType != '1d':
            heightEntry.insert(0, '5')
        else:
            heightEntry.insert(0, '3')
        widthEntry = tk.Entry(figureDimensionWindow,width=3)
        widthEntry.insert(0, '1')
        heightEntry.grid(row=1,column=1)
        widthEntry.grid(row=2,column=1)

        plotTitleWindow = tk.Frame(miscOptionsWindow)
        plotTitleWindow.grid(row=0,column=2,sticky=tk.N)
        plotTitleRadiobuttonList,plotTitleRadiobuttonVarsDict = ipe.createParameterSelectionRadiobuttons(plotTitleWindow,['plotTitle'],{'plotTitle':['yes','no']})

        legendWindow = tk.Frame(miscOptionsWindow)
        legendWindow.grid(row=0,column=3,sticky=tk.N)
        legendRadiobuttonList,legendRadiobuttonVarsDict = ipe.createParameterSelectionRadiobuttons(legendWindow,['legend'],{'legend':['yes','no']})

        colWrapWindow = tk.Frame(miscOptionsWindow)
        colWrapWindow.grid(row=0,column=4,sticky=tk.N)
        tk.Label(colWrapWindow,text='column wrap:').grid(row=0,column=0)
        colWrapEntry = tk.Entry(colWrapWindow,width=5)
        colWrapEntry.insert(0, '5')
        colWrapEntry.grid(row=1,column=0)

        titleAdjustWindow = tk.Frame(miscOptionsWindow)
        titleAdjustWindow.grid(row=0,column=5,sticky=tk.N)
        tk.Label(titleAdjustWindow,text='title location (% of window):').grid(row=0,column=0)
        titleAdjustEntry = tk.Entry(titleAdjustWindow,width=5)
        titleAdjustEntry.insert(0, '')
        titleAdjustEntry.grid(row=1,column=0)

        outlierWindow = tk.Frame(miscOptionsWindow)
        outlierWindow.grid(row=0,column=6,sticky=tk.N)
        outlierRadiobuttonList,outlierRadiobuttonVarsDict = ipe.createParameterSelectionRadiobuttons(outlierWindow,['remove outliers'],{'remove outliers':['yes','no']})
        outlierRadiobuttonVarsDict['remove outliers'].set('no')

        if subPlotType == 'kde':
            #Scale to mode button
            subPlotSpecificWindow = tk.Frame(self)
            subPlotSpecificWindow.pack(side=tk.TOP,pady=10)
            modeScaleWindow = tk.Frame(subPlotSpecificWindow)
            modeScaleWindow.grid(row=0,column=0,sticky=tk.N)
            modeScaleRadiobuttonList,modeScaleRadiobuttonVarsDict = ipe.createParameterSelectionRadiobuttons(modeScaleWindow,['scale to mode'],{'scale to mode':['yes','no']})
            modeScaleRadiobuttonVarsDict['scale to mode'].set('no')
            #Smoothness (or bandwidth) slider
            smoothnessWindow = tk.Frame(subPlotSpecificWindow)
            smoothnessWindow.grid(row=0,column=1,sticky=tk.N)
            smoothingSliderList = ipe.createParameterAdjustmentSliders(smoothnessWindow,['smoothing'],{'smoothing':[1,99,2,27]})

        plotButtonWindow = tk.Frame(self)
        plotButtonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(plotButtonWindow, text="Generate First Plot",command=lambda: collectInputs(False)).grid(row=0,column=0)
        tk.Button(plotButtonWindow, text="Generate All Plots",command=lambda: collectInputs(True)).grid(row=0,column=1)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: master.switch_frame(selectLevelsPage,experimentDf,homePage)).grid(row=len(scalingList)+4,column=0)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(assignLevelsToParametersPage,includeLevelValueList)).grid(row=len(scalingList)+4,column=1)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).grid(row=len(scalingList)+4,column=2)
