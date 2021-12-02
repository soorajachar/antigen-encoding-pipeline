# Cytokine processing pipeline
Pipeline to process cytokine data, extract integral features, train a neural network, and parameterize the latent space

To process the data, run the script named "antigen-encoding-pipeline-gui.py" (type ./antigen-encoding-pipeline-gui.py from the terminal)

When running for the first time, or when a new experiment needs to be added, drop the raw dataframes from plateypus in data/current/
Data must belong to one of these categories of experiment and are required to have the name of that category somewhere in their dataframe's name:
'PeptideComparison','TCellNumber','HighMI','Activation','DifferentAPC','Tumor','DifferentTCR','DrugPerturbation','hTCR','OT1CAR'

Then press:
1. Option 1 (Format raw dataframes)
2. Option 2 (Create or plot splines)
3. Select select the "create" option

To plot splines, press  
1. Option 2 (Create or plot splines)
2. Select the "plot" option. Only select datasets that have the same levels (error box will show if they do not)
3. Will appear in figures/splines

To create neural networks, press
1. Option 3 (Create and plot neural networks)
2. Follow prompts; remember to name your training dataset something meaningful; datasets will show under output/trained-networks
3. You will always be asked to plot your trained datasets; plots will show underneath figures/latent-spaces

To project datasets on trained neural networks:
1. Option 4 (Plot mutant projections on trained networks)
2. First select trained network to project with, then select dataset type to project on. WT datasets are training datasets. Plots will show under figures/latent-spaces

To parameterize datasets using a constant velocity or force fit:
1. Option 5 (Parameterize or plot latent spaces)
2. Choose training network to project with and dataset type to project on just like when plotting, then select fitType, and parameter (t0, v0 etc.) or parameterSpace (projection compared to fit) options
3. Plots will show under figures/parameter-spaces
4. Dataframes will show under output/parameter-dataframes or output/parameter-space-dataframes
