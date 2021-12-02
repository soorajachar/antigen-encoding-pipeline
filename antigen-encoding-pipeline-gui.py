#! /usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import os
import pickle,sys,subprocess
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
import seaborn as sns
sys.path.insert(0, 'scripts/process')
import adapt_dataframes
from process_raw_data import SplineDatasetSelectionPage
from neural_network import InputDatasetSelectionPage,import_WT_output
sys.path.insert(0, 'scripts/postprocess')
from latent_space import WeightMatrixSelectionPage
from parameterization import FittingFunctionSelectionPage
sys.path.insert(0, 'scripts/gui/plotting')
from plottingGUI import selectLevelsPage

def fileStructureCheck():
    for folder,subfolders in zip(['data','output','figures'],[['LOD','current','final','old','processed'],['parameter-dataframes','parameter-space-dataframes','trained-networks','projected-dataframes'],['latent-spaces','parameterized-spaces','splines']]):
        if folder not in os.listdir():
            subprocess.run(['mkdir',folder])
        for subfolder in subfolders:
            if subfolder not in os.listdir(folder):
                subprocess.run(['mkdir',folder+'/'+subfolder])

#Root class; handles frame switching in gui
class GUI_Start(tk.Tk):
    def __init__(self,startPage,*args):
        self.root = tk.Tk.__init__(self)
        self._frame = None
        self.homedirectory = os.getcwd()
        if self.homedirectory[-1] != '/':
            self.homedirectory+='/'
        self.homepage = startPage
        self.switch_frame(startPage,*args)

    def switch_frame(self, frame_class,*args):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self,*args)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

#Top level actions for cytokine processing gui 
class ActionSelectionPage(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self, master)
        mainWindow = tk.Frame(self)
        mainWindow.pack(side=tk.TOP,padx=10)
        
        l = tk.Label(mainWindow,text='Cytokine Processing GUI', font='Helvetica 18 bold')
        l.grid(row=0,column=0,columnspan=3)
        actionNames = ['Format raw dataframes','Create or plot splines','Create and plot trained neural networks','Plot mutant projections on trained network','Parameterize or plot latent spaces']
        actionVar = tk.StringVar(value=actionNames[0])
        rblist = []
        for i,actionName in enumerate(actionNames):
            rb = tk.Radiobutton(mainWindow, text=actionName,padx = 20, variable=actionVar, value=actionName)
            rb.grid(row=i+1,column=0,sticky=tk.W)
            rblist.append(rb)

        def collectInput():
            action = actionVar.get()
            if action == actionNames[0]:
                adapt_dataframes.main()
                tk.messagebox.showinfo("Info", "All raw dataframes formatted")
            elif action == actionNames[1]:
                with open('scripts/gui/plotting/plottingFolderName.pkl','wb') as f:
                    pickle.dump('splines',f)
                master.switch_frame(SplineDatasetSelectionPage)
            elif action == actionNames[2]:
                with open('scripts/gui/plotting/plottingFolderName.pkl','wb') as f:
                    pickle.dump('latent-spaces',f)
                master.switch_frame(InputDatasetSelectionPage)
            elif action == actionNames[3]:
                with open('scripts/gui/plotting/plottingFolderName.pkl','wb') as f:
                    pickle.dump('latent-spaces',f)
                latentSpaceBool = True
                master.switch_frame(WeightMatrixSelectionPage,latentSpaceBool,selectLevelsPage)
            elif action == actionNames[4]:
                with open('scripts/gui/plotting/plottingFolderName.pkl','wb') as f:
                    pickle.dump('parameterized-spaces',f)
                latentSpaceBool = False 
                master.switch_frame(WeightMatrixSelectionPage,latentSpaceBool,FittingFunctionSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,padx=10,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInput()).pack(side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(side=tk.LEFT)

def main():
    fileStructureCheck()
    app = GUI_Start(ActionSelectionPage)
    app.mainloop()

if __name__ == "__main__":
    main()
