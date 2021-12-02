#! /usr/bin/env python3
"""Project data in latent space of neural network"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle,sys
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
from scripts.gui.plotting.plottingGUI import selectLevelsPage
from scripts.process.adapt_dataframes import set_standard_order
from sklearn.preprocessing import MinMaxScaler

splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('antigen-encoding-pipeline')+1])+'/'

class WeightMatrixSelectionPage(tk.Frame):
    def __init__(self, master, lspB,nsp):
        tk.Frame.__init__(self, master)

        global latentSpaceBool,nextSwitchPage
        latentSpaceBool = lspB
        #Select levels page is nsp if we're starting at latent_space.py, Select Fit/action is nsp if starting at parameterization.py
        nextSwitchPage = nsp

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select trained weight matrix to project with:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        mainWindow = tk.Frame(self)
        datasetRadioButtons = []
        trainingDatasets = []
        print(path)
        for fileName in os.listdir(path+'output/trained-networks'):
            print(fileName)
            if '-' in fileName and '.pkl' in fileName and 'mlp' in fileName:
                datasetName = fileName.split('.')[0].split('-')[-1]
                if datasetName not in trainingDatasets:
                    trainingDatasets.append(datasetName)

        datasetVar = tk.StringVar(value=trainingDatasets[0])
        for i,dataset in enumerate(trainingDatasets):
            rb = tk.Radiobutton(mainWindow, text=dataset,padx = 20, variable=datasetVar,value=dataset)
            rb.grid(row=i,column=0,sticky=tk.W)
            datasetRadioButtons.append(rb)

        mainWindow.pack(side=tk.TOP,padx=10)

        def collectInputs():
            global datasetName
            datasetName = datasetVar.get()
            #Grab weight matrix
            mlp=pickle.load(open(path+'output/trained-networks/mlp-'+datasetName+'.pkl',"rb"))
            global weightMatrix
            weightMatrix = mlp
            #Grab training set min/max to normalize df to be projected
            global df_min,df_max
            df_min,df_max=pd.read_pickle(path+"output/trained-networks/min_max-"+datasetName+".pkl")
            master.switch_frame(WTorMutantDatasetSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(master.homepage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

class WTorMutantDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select projection type:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        mainWindow = tk.Frame(self)
        datasetRadioButtons = []
        #Mutant List
        mutantTypeList = ["Tumor","Activation","TCellNumber","DifferentAPC","OT1CAR","DifferentTCR","DrugPerturbation","HighMI","hTCR"]
        datasetTypes = ['WT']+mutantTypeList

        datasetVar = tk.StringVar(value=datasetTypes[0])
        for i,dataset in enumerate(datasetTypes):
            rb = tk.Radiobutton(mainWindow, text=dataset,padx = 20, variable=datasetVar,value=dataset)
            rb.grid(row=i,column=0,sticky=tk.W)
            datasetRadioButtons.append(rb)

        mainWindow.pack(side=tk.TOP,padx=10)

        def collectInputs():
            datasetType = datasetVar.get()
            if datasetType == 'WT':
                master.switch_frame(TrainingDatasetSelectionPage)
            else:
                #Load raw mutant data
                df_mutant = import_mutant_output(datasetType)
                #Subset features by what was used to train WT
                featureColumns = pd.read_pickle(path+"output/trained-networks/train-"+datasetName+".pkl").columns
                df_mutant = df_mutant.loc[:,featureColumns]
                print(df_mutant)
                #Normalize mutant
                df_mutant=(df_mutant - df_min)/(df_max - df_min)
                #dataList = []
                #for dataset in df_mutant.index.unique('Data'):
                #    dataDf = df_mutant.query("Data == @dataset")
                #    dataDf.loc[:,:] = MinMaxScaler().fit_transform(dataDf.values)
                #    dataList.append(dataDf)
                #df_mutant = pd.concat(dataList) 
                print(df_mutant)
                #Project mutant on latent space
                df_mutant_proj=pd.DataFrame(np.dot(df_mutant,weightMatrix.coefs_[0]),index=df_mutant.index,columns=["Node 1","Node 2"])
                projectionName = '-'.join([datasetName,datasetType])
                with open(path+'scripts/gui/plotting/projectionName.pkl','wb') as f:
                    pickle.dump(projectionName,f)
                df_mutant_proj = set_standard_order(df_mutant_proj.copy().reset_index(),returnSortedLevelValues=False)
                df_mutant_proj = pd.DataFrame(df_mutant_proj.iloc[:,-2:].values,index=pd.MultiIndex.from_frame(df_mutant_proj.iloc[:,:-2]),columns=['Node 1','Node 2'])
                with open(path+'output/projected-dataframes/projection-'+projectionName+'.pkl','wb') as f:
                    pickle.dump(df_mutant_proj,f)
                dataList = []
                for dataset in df_mutant.index.unique('Data'):
                    dataDf = df_mutant_proj.query("Data == @dataset")
                    if np.mean(dataDf['Node 1'].values) < 0:
                        dataDf['Node 1'] = dataDf['Node 1']*-1
                    dataList.append(dataDf)
                df_mutant_proj = pd.concat(dataList) 
                #switch to plotting
                if(latentSpaceBool):
                    proj_df = df_mutant_proj.iloc[::5,:]
                    master.switch_frame(nextSwitchPage,proj_df,WTorMutantDatasetSelectionPage)
                #switch to choosing fit type, then plotting
                else:
                    master.switch_frame(nextSwitchPage,df_mutant_proj,WTorMutantDatasetSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(WeightMatrixSelectionPage,latentSpaceBool,nextSwitchPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

class TrainingDatasetSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        labelWindow = tk.Frame(self)
        l1 = tk.Label(labelWindow, text="Select training dataset to project on:", font='Helvetica 18 bold').pack()
        labelWindow.pack(side=tk.TOP,padx=10,pady=10)

        datasetRadioButtons = []
        trainingDatasets = []
        all_df_WT = pd.read_pickle(path+"output/all-WT.pkl")
        idx = pd.IndexSlice
        sortedWT = set_standard_order(all_df_WT.loc[idx[:,'100k',:,:,1,'IFNg','concentration'],:].reset_index())
        for dataName in pd.unique(sortedWT['Data']):
            trainingDatasets.append(dataName)

        """BEGIN TEMP SCROLLBAR CODE"""
        labelWindow1 = tk.Frame(self)
        labelWindow1.pack(side=tk.TOP,padx=10,fill=tk.X,expand=True)

        #Make canvas
        w1 = tk.Canvas(labelWindow1, width=600, height=400, scrollregion=(0,0,3000,1000))

        #Make scrollbar
        scr_v1 = tk.Scrollbar(labelWindow1,orient=tk.VERTICAL)
        scr_v1.pack(side=tk.RIGHT,fill=tk.Y)
        scr_v1.config(command=w1.yview)
        #Add scrollbar to canvas
        w1.config(yscrollcommand=scr_v1.set)
        w1.pack(fill=tk.BOTH,expand=True)

        #Make and add frame for widgets inside of canvas
        #canvas_frame = tk.Frame(w1)
        mainWindow = tk.Frame(w1)
        mainWindow.pack()
        w1.create_window((0,0),window=mainWindow, anchor = tk.NW)
        """END TEMP SCROLLBAR CODE"""

        datasetVar = tk.StringVar(value=trainingDatasets[0])
        for i,dataset in enumerate(trainingDatasets):
            rb = tk.Radiobutton(mainWindow, text=dataset,padx = 20, variable=datasetVar,value=dataset)
            rb.grid(row=i,column=0,sticky=tk.W)
            datasetRadioButtons.append(rb)

        def collectInputs():
            datasetName2 = datasetVar.get()
            #Load files
            df_WT = pd.concat([all_df_WT.loc[datasetName2]],keys=[datasetName2],names=['Data'])
            df_WT = df_WT.unstack(['Feature','Cytokine']).loc[:,'value']
            #df_WT=pd.read_pickle(path+"output/trained-networks/train-"+datasetName2+".pkl")
            #Subset features by what was used to train WT
            featureColumns = pd.read_pickle(path+"output/trained-networks/train-"+datasetName+".pkl").columns
            df_WT = df_WT.loc[:,featureColumns]
            print(df_WT)
            #Normalize WT
            df_WT=(df_WT - df_min)/(df_max - df_min)
            #Project WT on latent space
            df_WT_proj=pd.DataFrame(np.dot(df_WT,weightMatrix.coefs_[0]),index=df_WT.index,columns=["Node 1","Node 2"])
            #switch to plotting
            projectionName = '-'.join([datasetName,datasetName2])
            with open(path+'scripts/gui/plotting/projectionName.pkl','wb') as f:
                pickle.dump(projectionName,f)
            df_WT_proj = set_standard_order(df_WT_proj.copy().reset_index(),returnSortedLevelValues=False)
            df_WT_proj = pd.DataFrame(df_WT_proj.iloc[:,-2:].values,index=pd.MultiIndex.from_frame(df_WT_proj.iloc[:,:-2]),columns=['Node 1','Node 2'])
            with open(path+'output/projected-dataframes/projection-'+projectionName+'.pkl','wb') as f:
                pickle.dump(df_WT_proj,f)
            if(latentSpaceBool):
                proj_df = df_WT_proj.iloc[::5,:]
                master.switch_frame(nextSwitchPage,proj_df,TrainingDatasetSelectionPage)
            #switch to choosing fit type, then plotting
            else:
                master.switch_frame(nextSwitchPage,df_WT_proj,TrainingDatasetSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(WTorMutantDatasetSelectionPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

def import_WT_output():
    """Import splines from wildtype naive OT-1 T cells by looping through all datasets
    Returns:
            df_full (dataframe): the dataframe with processed cytokine data
    """

    folder="../data/processed/"

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
            "DrugAdditionTime":24,
            "Drug":"Null"
            }

    for file in os.listdir(folder):

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

def import_mutant_output(mutant):
    """Import processed cytokine data from an experiment that contains mutant data

    Args:
            mutant (str): name of file with mutant data.
                    Has to be one of the following "Tumor","Activation","TCellNumber","DifferentAPC","OT1CAR","DifferentTCR"

    Returns:
        df_full (dataframe): the dataframe with processed cytokine data
    """

    naive_level_values={
                "ActivationType": "Naive",
                "Antibody": "None",
                "APC": "B6",
                "APCType": "Splenocyte",
                "CARConstruct":"None",
                "ConditionType":"Coculture",
                "CAR_Antigen":"None",
                "Genotype": "WT",
                "IFNgPulseConcentration":"None",
                "TCellType": "OT1",
                "TLR_Agonist":"None",
            }

    mutant_levels={
                "Tumor": ["APC","APCType","IFNgPulseConcentration"],
                "Activation": ["ActivationType"],
                "TCellNumber": [],
                "DifferentAPC": ["TLR_Agonist","APCType"],
                "OT1CAR":["CAR_Antigen","Genotype","CARConstruct"],
                "DifferentTCR":["TCellType"],
                "DrugPerturbation":['DrugAdditionTime','Drug'],
                "HighMI":['Replicate'],
                "hTCR":['Donor'],
            }

    essential_levels=["TCellNumber","Peptide","Concentration","Time"]

    folder=path+"data/processed/"

    for file in os.listdir(folder):

        if (mutant not in file) | (".hdf" not in file):
            continue

        df=pd.read_hdf(folder + "/" + file)

        # If level not in essential levels or mutant-required level, keep naive level values and drop level
        print(file)
        print(df.index.names)
        for level in df.index.names:
            if level not in essential_levels+mutant_levels[mutant]:
                df=df[df.index.get_level_values(level)==naive_level_values[level]]
                df=df.droplevel(level,axis=0)

        df=pd.concat([df],keys=[file[:-4]],names=["Data"]) #add experiment name as multiindex level

        print(file)
        print(df.index.names)

        if "df_full" not in locals():
            df_full=df.copy()
        else:
            df_full=pd.concat([df_full,df],levels=["Data"]+mutant_levels[mutant]+essential_levels)

    return df_full
