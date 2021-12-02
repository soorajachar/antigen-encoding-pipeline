#! /usr/bin/env python3
"""Train and save a neural network"""

import os
import pickle
import sys
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
	import matplotlib
	matplotlib.use("TkAgg")
from sklearn import neural_network
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from scripts.gui.plotting.plottingGUI import createLabelDict,checkUncheckAllButton,selectLevelsPage
from scripts.process.adapt_dataframes import set_standard_order

splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('antigen-encoding-pipeline')+1])+'/'

idx = pd.IndexSlice

#FOR NEURAL NETWORK DATA SELECTION
class InputDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        #Import data
        df = import_WT_output()
        dataset = df.stack().stack().to_frame('value')
        dataset.to_pickle(path+"output/all-WT.pkl")

        tk.Frame.__init__(self, master)

        global trueLabelDict
        trueLabelDict = {}

        #Sort by date/number/quality/quantity
        print(dataset)
        dataset = set_standard_order(dataset.reset_index())
        sortedValues = set_standard_order(dataset.copy(),returnSortedLevelValues=True)

        dataset = pd.DataFrame(dataset['value'].values,index=pd.MultiIndex.from_frame(dataset.iloc[:,:-1]),columns=['value'])
        trueLabelDict = createLabelDict(dataset.copy(),sortedValues=sortedValues)

        titleWindow = tk.Frame(self)
        titleWindow.pack(side=tk.TOP,padx=10,fill='none',expand=True)
        titleLabel = tk.Label(titleWindow, text='Training set name:',pady=10, font='Helvetica 18 bold').grid(row=0,column = 0)
        e1 = tk.Entry(titleWindow)
        e1.grid(row=0,column=1)
        e1.insert(0, 'default')

        timeWindow = tk.Frame(self)
        timeWindow.pack(side=tk.TOP,padx=10,fill='none',expand=True)
        timeLabel = tk.Label(timeWindow, text='Time range (START TIME-END TIME):',pady=10, font='Helvetica 18 bold').grid(row=0,column = 0)
        e2 = tk.Entry(timeWindow)
        e2.grid(row=0,column=1)
        e2.insert(0, '1-72')

        """BEGIN TEMP SCROLLBAR CODE"""
        labelWindow1 = tk.Frame(self)
        labelWindow1.pack(side=tk.TOP,padx=10,fill=tk.X,expand=True)

        #Make canvas
        w1 = tk.Canvas(labelWindow1, width=1500, height=600, scrollregion=(0,0,3000,1200))

        #Make scrollbar
        scr_v1 = tk.Scrollbar(labelWindow1,orient=tk.VERTICAL)
        scr_v1.pack(side=tk.RIGHT,fill=tk.Y)
        scr_v1.config(command=w1.yview)
        #Add scrollbar to canvas
        w1.config(yscrollcommand=scr_v1.set)
        w1.pack(fill=tk.BOTH,expand=True)

        #Make and add frame for widgets inside of canvas
        #canvas_frame = tk.Frame(w1)
        labelWindow = tk.Frame(w1)
        labelWindow.pack()
        w1.create_window((0,0),window=labelWindow, anchor = tk.NW)
        """END TEMP SCROLLBAR CODE"""
        #labelWindow = tk.Frame(self)
        #labelWindow.pack(side=tk.TOP,padx=10,fill=tk.X,expand=True)

        l1 = tk.Label(labelWindow, text='Parameters:',pady=10, font='Helvetica 18 bold').grid(row=0,column = 0,columnspan=len(trueLabelDict)*6)
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

        def collectInputs(dataset):
            includeLevelValueList = []
            #Decode boolean array of checkboxes to level names
            i = 0
            for levelName,checkButtonVariableList in zip(trueLabelDict,overallCheckButtonVariableList):
                tempLevelValueList = []
                for levelValue,checkButtonVariable in zip(trueLabelDict[levelName],checkButtonVariableList):
                    if checkButtonVariable.get():
                        tempLevelValueList.append(levelValue)
                #Add time level values in separately using time range entrybox
                if i == len(trueLabelDict.keys()) - 2:
                    timeRange = e2.get().split('-')
                    includeLevelValueList.append(list(range(int(timeRange[0]),int(timeRange[1]))))
                includeLevelValueList.append(tempLevelValueList)
                i+=1

            print(dataset)
            df = dataset.loc[tuple(includeLevelValueList),:].unstack(['Feature','Cytokine']).loc[:,'value']
            print(df)

            trainingSetName = e1.get()
            #Save min/max and normalize data
            pickle.dump([df.min(),df.max()],open(path+"output/trained-networks/min_max-"+trainingSetName+".pkl","wb"))
            df=(df - df.min())/(df.max()-df.min())
            df.to_pickle(path+"output/trained-networks/train-"+trainingSetName+".pkl")

            #Needed to adapt this to work with any selection of peptides; not sure if what I'm doing here is correct
            peptides=["N4","Q4","T4","V4","G4","E1"][::-1]
            full_peptide_dict={k:v for v,k in enumerate(peptides)}
            peptide_dict = {}
            for peptide in full_peptide_dict:
                if peptide in pd.unique(df.index.get_level_values("Peptide")):
                    peptide_dict[peptide] = full_peptide_dict[peptide]

            #Extract times and set classes
            y=df.index.get_level_values("Peptide").map(peptide_dict)

            mlp=neural_network.MLPClassifier(
                    activation="tanh",hidden_layer_sizes=(2,),max_iter=5000,
                    solver="adam",random_state=90,learning_rate="adaptive",alpha=0.01).fit(df,y)

            score=mlp.score(df,y); print("Training score %.1f"%(100*score));
            pickle.dump(mlp,open(path+"output/trained-networks/mlp-"+trainingSetName+".pkl","wb"))

            df_WT_proj=pd.DataFrame(np.dot(df,mlp.coefs_[0]),index=df.index,columns=["Node 1","Node 2"])
            df_WT_proj.to_pickle(path+"output/trained-networks/proj-WT-"+trainingSetName+".pkl")

            proj_df = df_WT_proj.iloc[::5,:]
            master.switch_frame(selectLevelsPage,proj_df,InputDatasetSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)

        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs(dataset)).grid(row=maxNumLevelValues+4,column=0)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(master.homepage)).grid(row=maxNumLevelValues+4,column=1)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).grid(row=maxNumLevelValues+4,column=2)

def import_WT_output():
    """Import splines from wildtype naive OT-1 T cells by looping through all datasets

    Returns:
            df_full (dataframe): the dataframe with processed cytokine data
    """

    folder=path+"data/processed/"

    naive_pairs={
            "ActivationType": "Naive",
            "Antibody": "None",
            "APC": "B6",
            "APCType": "Splenocyte",
            "CARConstruct":"None",
            "CAR_Antigen":"None",
            "Genotype": "WT",
            "IFNgPulseConcentration":"None",
            "TCellType": "OT1",
            "TLR_Agonist":"None",
            "TumorCellNumber":"0k",
            "DrugAdditionTime":36,
            "Drug":"Null"
            }
    validnames = ['TCellNumber','PeptideComparison','Activation','PeptideTumor']
    for file in os.listdir(folder):
        valid = False
        for validname in validnames:
            if validname in file:
                valid = True
                break
        if valid:
            print(file)
            if ".hdf" not in file:
                continue

            df=pd.read_hdf(folder + file)
            mask=[True] * len(df)

            for index_name in df.index.names:
                if index_name in naive_pairs.keys():
                    mask=np.array(mask) & np.array([index == naive_pairs[index_name] for index in df.index.get_level_values(index_name)])
                    df=df.droplevel([index_name])

            df=pd.concat([df[mask]],keys=[file[:-4]],names=["Data"]) #add experiment name as multiindex level
            if "df_full" not in locals():
                df_full=df.copy()
            else:
                df_full=pd.concat((df_full,df))
    return df_full

def plot_weights(mlp,cytokines,peptides,**kwargs):

    fig,ax=plt.subplots(1,2,figsize=(8,2))
    ax[0].plot(mlp.coefs_[0],marker="o",lw=2,ms=7)
    ax[1].plot(mlp.coefs_[1].T[::-1],marker="o",lw=2,ms=7)
    [a.legend(["Node 1","Node 2"]) for a in ax]

    ax[0].set(xlabel="Cytokine",xticks=np.arange(len(cytokines)),xticklabels=cytokines,ylabel="Weights")
    ax[1].set(xlabel="Peptide",xticks=np.arange(len(mlp.coefs_[1].T)),xticklabels=peptides)
    plt.savefig("%s/weights.pdf"%tuple(kwargs.values()),bbox_inches="tight")
