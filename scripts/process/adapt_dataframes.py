#! /usr/bin/env python3
"""Adapt dataframes to processing pipeline

- Update filenames, level names and level values
- Retrieve standard ordering for T cell number, peptide and concentration

Filename changes
        TCellTypeComparison_OT1,P14,F5_Timeseries_3
                No F5 in this dataset, remove mention of F5 in filename

        NaiveVsExpandedTCells_OT1_Timeseries_1
                Change filename to Activation_Timeseries_1

        CD4_5CC7_2
                Change filename to TCellType_OT1_CD4_5CC7_Timeseries_2

        AntagonismComparison_OT1_Timeseries_1
                Change filename to OT1_Antagonism_1

Level name changes
        APC_Type, APC Type --> APCType
        Agonist, TCR_Antigen --> Peptide

Level value changes
        Activation_TCellNumber_1
                TCellType --> ActivationType

        Tumor datasets
                Include IFNg pulse concentration with Nones for Splenocytes
                Adapt concentrations
                Change TCell/TumorCellNumber format
                Add APC & APCType level TODO: APC is B16?

        CAR datasets
                Genotype: Naive --> WT
                Concentration: 1uM

        ITAMDef datasets
                Include TCellNumber (30k, not standard 100k)

        TCellType datasets
                Peptide2: NotApplicable --> None

        Antagonism datasets
                Select nonantagonist data
"""

import warnings
warnings.filterwarnings("ignore")

import os, pickle, sys, re
import numpy as np
import pandas as pd

splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('antigen-encoding-pipeline')+1])+'/'
folder=path+"data/current/"

filename_changes={
                                        "TCellTypeComparison_OT1,P14,F5_Timeseries_3": "TCellTypeComparison_OT1,P14_Timeseries_3",
                                        "NaiveVsExpandedTCells_OT1_Timeseries_1": "Activation_Timeseries_1",
                                        "CD4_5CC7_2":"TCellType_OT1_5CC7_Timeseries_2",
                                        "PeptideComparison_OT1_Timeseries_20":"NewPeptideComparison_OT1_Timeseries_20"
                                        }

level_name_changes={
                                        "APC Type":"APCType",
                                        "APC_Type":"APCType",
                                        "Agonist":"Peptide",
                                        "TCR_Antigen":"Peptide"
                                        }

tumor_timeseries = [
                                        "TumorTimeseries_1",
                                        "TumorTimeseries_2",
                                        "PeptideTumorComparison_OT1_Timeseries_1",
                                        "PeptideTumorComparison_OT1_Timeseries_2"
                                        ]

activation_timeseries = ["Activation_TCellNumber_1"]

itamdef_timeseries = ["ITAMDeficient_OT1_Timeseries_%d"%num for num in [9,10,11]]

tcelltype_timeseries = ["TCellTypeComparison_OT1,P14,F5_Timeseries_3"]

def sort_SI_column(columnValues,unitSuffix):
    si_prefix_dict = {'a':1e-18,'f':1e-15,'p':1e-12,'n':1e-9,'u':1e-6,'m':1e-3,'':1e0}
    numericValues = []
    for val in columnValues:
        val = val.replace(' ','')
        splitString = re.split('(\d+)',val)[1:]
        #If no numeric variables, assume zero
        if len(splitString) < 2:
            numeric_val = 0
        else:
            #If no numeric variables, assume zero
            if splitString[1] == '':
                numeric_val = 0
            else:
                #If correctly SI formatted, use si prefix dict
                if len(splitString[1]) < 3:
                    siUnit = splitString[1].split(unitSuffix)[0]
                #If strange unit, like ug/mL, just assign lowest SI prefix to shunt to the end of the order
                else:
                    siUnit = 'a'
                numeric_val = si_prefix_dict[siUnit]*float(splitString[0])
        numericValues.append(numeric_val)
    return numericValues

def return_data_date_dict():
    splitPath = os.getcwd().split('/')
    rootIndex = splitPath.index('antigen-encoding-pipeline')
    rootDir = '/'.join(splitPath[:rootIndex+1])+'/'
    dataFolder = rootDir+'data/final'
    dateDict = {}
    for fileName in os.listdir(dataFolder):
        if '.pkl' in fileName:
            datasetDate = fileName.split('-')[1]
            datasetName = fileName.split('-')[2]
            dateDict[datasetName] = float(datasetDate)*-1
    return dateDict

def set_standard_order(df,returnSortedLevelValues=False):
        levelDict = {'DATASORT':'Data','TCELLSORT':'TCellNumber','PEPTIDESORT':'Peptide','CONCSORT':'Concentration'}
        peptide_dict={"N4":13,"A2":12,"Y3":11,"Q4":10,"T4":9,"V4":8,"G4":7,"E1":6,
                                        "AV":5,"A3V":4,"gp33WT":3,"None":2}
        levelsToSort = []
        if 'Data' in df.columns:
            dataset_dict = return_data_date_dict()
            df["DATASORT"]=df.Data.map(dataset_dict)
            levelsToSort.append('DATASORT')
        if 'TCellNumber' in df.columns:
            df["TCELLSORT"]=df.TCellNumber.str.replace("k","").astype("float")
            levelsToSort.append('TCELLSORT')
        if 'Peptide' in df.columns:
            df["PEPTIDESORT"]=df.Peptide.map(peptide_dict)
            levelsToSort.append('PEPTIDESORT')
        if 'Concentration' in df.columns:
            df["CONCSORT"]=sort_SI_column(df.Concentration,'M')
            levelsToSort.append('CONCSORT')
        sortColumnRemovalVar = -1*len(levelsToSort)

        if not returnSortedLevelValues:
            df = df.sort_values(levelsToSort,ascending=False).iloc[:,:sortColumnRemovalVar]
            return df
        else:
            sortedLevelValues = {}
            for levelToSort in levelsToSort:
                levelValues = list(pd.unique(df.sort_values([levelToSort],ascending=False).loc[:,levelDict[levelToSort]]))
                sortedLevelValues[levelDict[levelToSort]] = levelValues
            return sortedLevelValues

def main():

        for file in os.listdir(folder):
            if '.pkl' in file or '.hdf' in file:
                if '.pkl' in file:
                    tmp = pd.read_pickle(folder+file)
                elif '.hdf' in file:
                    tmp = pd.read_hdf(folder+file,key='df')
                #filename=file[41:-13]
                filename=file.split('-')[2]

                # Change orientation of dataframe
                df=tmp.stack(["Time"])

                # Then remove levels for simplified manipulation of level values
                df=df.reset_index()

                # Change levelnames if one of the levels occur
                for idx,level_name in enumerate(df.columns):
                        if level_name in level_name_changes.keys():
                                df.columns = list(df.columns[:idx])+[level_name_changes[level_name]]+list(df.columns[idx+1:])

                print(filename)
                new_file = file.split('.')[0]+"-final.pkl"

                # Change filename
                #if filename in filename_changes.keys():
                #        new_file=file[:41]+filename_changes[file[41:-13]]+"-final.pkl"
                # Add IFNg pulse concentration and tumor characteristics
                if filename in tumor_timeseries[:2]:
                        df["IFNgPulseConcentration"]="None"
                        df["IFNgPulseConcentration"][df.APCType=="Tumor"]=[conc for conc in df.Concentration[df.APCType=="Tumor"]]
                        df["Concentration"][df.APCType=="Tumor"]="None"
                elif filename in tumor_timeseries[2]:
                        df["APC"]=["B16" if apctype == "Tumor" else "B6" for apctype in df.APCType]
                        df["IFNgPulseConcentration"]=[conc[:3] if "IFNg" in conc else "None" for conc in df.Concentration]
                        df["Concentration"]=["None" if "IFNg" in conc else conc for conc in df.Concentration]

                elif filename in tumor_timeseries[3]:
                        df["APC"]="B16"
                        df["APCType"]="Tumor"
                        df["Concentration"]="None"
                        df["TCellNumber"]=df.TCellNumber.str.replace("K","k")
                        df["TumorCellNumber"]=df.TumorCellNumber.str.replace("K","k")
                        df=df[df.TumorCellNumber=="17k"]
                        df.drop("TumorCellNumber",axis=1,inplace=True)
                        df=df.iloc[:,::-1] # Hacky shortcut to get ordering of columns that results in similar level order for APCType, APC, IFNgPulseConcentration

                # Add activation type
                elif filename in activation_timeseries:
                        df["ActivationType"]="Naive"
                        df["ActivationType"][df.TCellType=="Blast"]="aCD3_aCD28"
                        df.drop("TCellType",axis=1,inplace=True)

                # Add concentration level for CAR timeseries
                elif "CAR" in filename:
                        df["CARConstruct"]=df.Genotype
                        df["CARConstruct"][df.CARConstruct=="Naive"]="None"
                        df.drop("Genotype",axis=1,inplace=True)
                        df["Concentration"]="1uM"

                # ITAMDefs after May have 30k T cells
                elif ("ITAMDef" in filename) and not filename.endswith("8"):
                        df["TCellNumber"]="30k"

                # Make Peptide2 level name compatible with None
                elif filename in tcelltype_timeseries:
                        df["Peptide2"]=df.Peptide2.str.replace("NotApplicable","None")
                        df=df[df["Peptide2"]=="None"]
                        df.drop("Peptide2",axis=1,inplace=True)

                # TODO swap A2/Y3 in Pepcomp20

                # Remove nonWT antagonism data
                # TODO retrieve lost data by treating the duplicate entries separataly
                # [(Experiment=None)&(Antagonist=None),(Agonist_Antagonist_Locations=Blank)]
                elif "Antagonism" in filename:

                        if "TCellType" in df.columns:
                                df=df[(df.Peptide == "None") | (df.Peptide2 == "None")]
                                df["Peptide"]=(df["Peptide"]+df["Peptide2"]).str.replace("None","")
                                df["Peptide"][df.Peptide == ""]="None"
                                df["TCellType"]= ["P14" if peptide in ["A3V","AV","gp33WT"] else "OT1" for peptide in df.Peptide]
                                df=df[["Cytokine","TCellType","Peptide","Concentration","Time",0]]

                        else:

                                if "Experiment" in df.columns:
                                        df=df[df.Experiment == "Calibration"]
                                elif "Agonist_Antagonist_Locations" in df.columns:
                                        df=df[df.Agonist_Antagonist_Locations == "NotApplicable"]
                                else:
                                        df=df[df.Antagonist == "None"]
                                df=df[["Cytokine","Peptide","Concentration","Time",0]]

                # Add TCellNumber level
                if "TCellNumber" not in df.columns:
                        df["TCellNumber"]="100k"

                df=set_standard_order(df)

                # Set (possible new) columns as index levels except for the column with concentrations called 0
                # Place standard levels last (TCellNumber/Peptide/Concentration)
                df.set_index([col for col in df.columns if col is not 0],inplace=True)

                standard_levels=["TCellNumber","Peptide","Concentration"]
                df=df.reorder_levels([level for level in df.index.names if level not in standard_levels]+standard_levels)

                # Revert to original orientation of dataframe
                df=df.unstack("Time").droplevel(level=0,axis=1)

                # Print changes
                if df.index.names != tmp.index.names:
                        None
                        # print(file)
                        # print("OLD\t",tmp.index.names)
                        # print("NEW\t",df.index.names,"\n")

                # Save file
                df.to_pickle(path+"data/final/"+new_file)


if __name__ == "__main__":
        main()
