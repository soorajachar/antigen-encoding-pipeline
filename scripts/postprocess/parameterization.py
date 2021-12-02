#! /usr/bin/env python3
"""Fit curves in latent space"""

import warnings
warnings.filterwarnings("ignore")

import pickle,sys,os
import numpy as np
import pandas as pd
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

#from scipy.optimize import curve_fit  # Use our custom curve_fit instead
from scripts.postprocess.fitting_functions import curve_fit_jac
from scipy.special import hyp2f1
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
from scripts.gui.plotting.plottingGUI import selectLevelsPage
from scripts.postprocess.latent_space import WeightMatrixSelectionPage
from scripts.process.adapt_dataframes import set_standard_order

splitPath = os.getcwd().split('/')
path = '/'.join(splitPath[:splitPath.index('antigen-encoding-pipeline')+1])+'/'

def fit_vt_vm(node1, node2, tol_r2=0.99):
    """Fit a straight line to the final time points of latent space.
    Add time points until R^2 falls below tol_r2, when the curve is no longer straight.
    Requires at least 5 timepoints for a good fit, else returns NaN

    Args:
        node1 (1darray): timepoints of node 1
        node2 (1darray): timepoints of node 2
        tol_r2 (float): tolerance of the fit

    Returns:
        slope (float): slope of the fit (vt/vm ratio) or NaN when the curve could not be fit
    """

    def check_uniqueness(n_points=1):
        """Recursively find number of points to start fitting with more than one unique value in Node 1 or Node 2

        Args:
            n_points (int): default is 1

        Returns:
            n_points (int): number of points to start fit with
        """

        if ((len(np.unique(node1[-n_points:])) == 1) and (len(np.unique(node2[-n_points:])) == 1)):
            if len(node1) < n_points:
                return n_points
            else:
                return check_uniqueness(n_points+1)

        else:
            return n_points

    n_points = check_uniqueness(4)
    if len(node1) < n_points:
        return np.nan
    r2 = 1

    while (r2 > tol_r2) and (n_points < len(node1)):
        slope,_,r2,_,_ = linregress(node1[-n_points:], node2[-n_points:])
        n_points += 1

    if ((n_points >= 5) and (np.abs(slope) > 1)):
        return slope
    else:
        return np.nan

def compute_vt_vm(df,slope_type="median"):
    """ Compute vt/vm ratio.
    Fit all conditions separately and return the mean or median slope (whichever is better)

    Args:
        df (pd.DataFrame): dataframe with all conditions
        slope_type (string): determine measure to output slope from distribution.
            Currently implemented measures are mean or median. Default is median.

    Returns:
        slope (int): mean or median slope corresponding to vt/vm ratio

    """
    slopes = pd.DataFrame([], index=df.index.droplevel("Time").unique(), columns=["slope"])
    # Fit a straight line to as as many final points as possible (at least 5)
    for idx in slopes.index:
        slopes.loc[idx] = fit_vt_vm(df.loc[idx,"Node 1"], df.loc[idx,"Node 2"])
    # Return mean or median dropping conditions for which no slope could be reliably calculated (n_points <= 5)
    median_slope = slopes.dropna().median().values[0]
    mean_slope = slopes.dropna().mean().values[0]
    print("\tMedian slope:\t",np.around(median_slope,2))
    print("\tMean slope:\t",np.around(mean_slope,2))

    if slope_type == "median":
        return median_slope
    elif slope_type == "mean":
        return mean_slope
    else:
        raise NotImplementedError("%s measure is not yet implemented"%slope_type)


def ballistic_v0(times, v0, t0, theta, vt):
    """Piecewise ballisfic function that computes node 1 and node 2 from given parameters.
    Phase 1 is constant velocity regime.

    Args:
        times (1darray): timepoints at which to compute the ballistic trajectories.
        v0 (float): constant speed
        t0 (float): time until phase 1
        theta (float): angle of constant speed (radians)
        vt (float): terminal velocity

    Returns:
        curves (2darray): array with node 1 and node 2, shape [2, times]
            Our custom curve_fit_jac can deal with vector-valued functions.
    """
    # Initialize some variables
    vm = vt / vt_vm_ratio

    x = np.zeros(times.shape)
    y = np.zeros(times.shape)
    vx = v0 * np.cos(theta-np.pi/2)
    vy = v0 * np.sin(theta-np.pi/2)

    # Phase 1
    prop = (times <= t0)
    x[prop] = vx * times[prop]
    y[prop] = vy * times[prop]

    # Phase 2
    r0 = [vx*t0,vy*t0]  # Position at the end of the propulsion phase
    delta_t = times[~prop] - t0

    x[~prop] = (vx + vm) * (1-np.exp(-2*delta_t))/2 - vm * delta_t + r0[0]
    y[~prop] = (vy + vt) * (1-np.exp(-2*delta_t))/2 - vt * delta_t + r0[1]

    return np.array([x, y])


def ballistic_F(times, F,  t0, theta, vt):
    """Piecewise ballisfic function that computes node 1 and node 2 from given parameters.
    Phase 1 is constant force regime.

    Args:
        times (1darray): timepoints at which to compute the ballistic trajectories.
        F (float): constant speed
        t0 (float): time until phase 1
        theta (float): angle of constant speed
        vt (float): terminal velocity

    Returns:
        curves (2darray): array with node 1 and node 2, shape [2, times]
            Our custom curve_fit_jac can deal with vector-valued functions.
    """

    def phase1_position(F,t):
        return F * (t - 1 + np.exp(-t))

    def phase1_velocity(F,t):
        return F * (1 - np.exp(-t))

    # Initialize some variables
    vm = vt / vt_vm_ratio

    x = np.zeros(times.shape)
    y = np.zeros(times.shape)
    Fx = F * np.cos(theta-np.pi/2)# - vm
    Fy = F * np.sin(theta-np.pi/2)# - vt

    # Phase 1
    prop = (times <= t0)
    x[prop] = phase1_position(Fx,times[prop])
    y[prop] = phase1_position(Fy,times[prop])

    # Position and velocity at end of phase 1
    r0 = phase1_position(np.array([Fx,Fy]),t0)
    v0 = phase1_velocity(np.array([Fx,Fy]),t0)

    # Phase 2
    delta_t = times[~prop] - t0

    x[~prop] = (v0[0] + vm) * (1-np.exp(-delta_t)) - vm * delta_t + r0[0]
    y[~prop] = (v0[1] + vt) * (1-np.exp(-delta_t)) - vt * delta_t + r0[1]

    return np.array([x,y])

## Improved ballistic model with a sigmoid and fixed alpha.
# Full  model for sigmoid concentration n1 and n2 given just above.
def sigmoid_conc_full(tau, params):
    """ Argument t can be an array, other parameters are scalars. params is a list. """
    # Extract parameters
    a0, t0, theta, v1, gamma = params
    a1, a2 = a0 * np.cos(theta), a0 * np.sin(theta)  # theta in radians.

    # Vector where 1st dimension is n1, second is n2
    r = np.zeros((2, t.shape[0]))

    # Common terms
    bound_exp = 1 - np.exp(-tau)
    exp_beta = np.exp(gamma*(tau - t0)) + 1

    # Node 1
    r[0] = bound_exp * ((a1 + v1)/exp_beta - v1)

    # Node 2
    r[1] = (a2 + v2) * np.power(bound_exp, 2) / exp_beta - bound_exp * v2

    return r

# The integral I(\tau, \tau_0, \gamma) = \int \frac{- e^{-\tau}}{e^{\gamma(\tau - \tau_0)} + 1}
# general and special gamma cases covered.
def sig_int(tau, tau_0, gamma):
    """ Integral of -e^{-\tau} / (1 + e^{\gamma(\tau - \tau_0)})
    """
    # Verification: is gamma a special case where the hypergeometric function fails?
    if abs(int(1/gamma) - 1/gamma) < 1e-12 and abs(int(1/gamma)) > 0.01:  # Not the 1/gamma = 0 case.
        print("We hit the special case 1/gamma = {}".format(1/gamma))
        n = int(1/gamma)
        res = np.exp(-tau)
        res += (-1)**n * n * np.exp(-tau_0) * np.log(np.exp(-gamma * tau) + np.exp(-gamma * tau_0))
        # Simpler to code a Python loop than to create a 2d array with one row per term in the sum...
        # If performance really becomes an issue, create arrays. But will probably never use this.
        for j in range(1, n):  # from 1 to n-1 included
            res += (-1)**j * n / (n-j) * np.exp(-j * tau_0 / n) * np.exp((j/n - 1) * tau)

    # Otherwise, can rely on 2F1
    else:
        res = np.exp(-tau) * hyp2f1(1, -1/gamma, 1 - 1/gamma, -np.exp(gamma*(tau - tau_0)))

    return res

# The main function computing [N_1, N_2]
def ballistic_sigmoid(tau, a0, t0, theta, v1, gamma):
    """ Integral expression for N_1, N_2, based on the improved ballistic equations
    with a sigmoid for n_1 and n_2.

    Args:
        times (1darray): timepoints at which to compute the ballistic trajectories.
        a0 (float): initial acceleration
        t0 (float): time of half-maximum sigmoidal decay
        theta (float): angle of initial velocity (radians)
        v1 (float): terminal concentration in node 1
        gamma (float): degradation time scale, relative to alpha (beta/alpha)

    Returns:
        curves (2darray): array with node 1 and node 2, shape [2, times]
            Our custom curve_fit_jac can deal with vector-valued functions.
    """
    # Initialize some variables
    a1, a2 = a0 * np.cos(theta), a0 * np.sin(theta)
    v2 = v1 * vt_vm_ratio  # vt is v2 (y direction), vm is v1 (x direction)

    # Some terms used in both node 1 and 2
    lnterm_0 = np.log(1 + np.exp(-gamma*t0)) / gamma
    lnterm = np.log(np.exp(-gamma*tau) + np.exp(-gamma*t0)) / gamma
    boundedint = tau + np.exp(-tau)
    siginterm = sig_int(tau, t0, gamma)

    # Constants to ensure it's zero at the origin
    k1 = v1 - (a1 + v1) * (sig_int(0, t0, gamma) - lnterm_0)
    k2 = v2 + (a2 + v2) * (sig_int(0, 2*t0, gamma/2)/2 - 2*sig_int(0, t0, gamma) + lnterm_0)

    # Assemble terms
    N1 = (a1 + v1) * (siginterm - lnterm) - v1 * boundedint + k1
    N2 = (a2 + v2) * (2*siginterm - sig_int(2*tau, 2*t0, gamma/2)/2 - lnterm)
    N2 += -v2 * boundedint + k2

    return np.array([N1, N2])

## Improved ballistic model with a sigmoid and free alpha (more parameters)
## TODO!


### FUNCTIONS PERFORMING THE FIT FOR EACH TIME COURSE
# Find the best fit for each time course in the DataFrame.
def return_fit_params(df, func, bounds, p0, param_labels, time_scale=20,
                        reg_rate=None, offsets=None, func_kwargs={}):
    # Initialize a dataframe that will record parameters and parameter variances fitted to each curve.
    var_param_labels = ["var" + param for param in param_labels]
    cols = param_labels+var_param_labels
    nparams = len(param_labels)

    df_hess = pd.DataFrame([], index=df.index.droplevel("Time").unique(),
            columns=pd.MultiIndex.from_product([param_labels, param_labels]))
    df_params = pd.DataFrame([], index=df.index.droplevel("Time").unique(), columns=cols)

    for idx in df_params.index:
        # With the custom curve_fit_jac, no need to tile for regularization;
        # it is included as a curve_fit parameter and automatically taken care of.
        # So it's easier to define a new function, returning vector values at each xdata
        ydata = df.loc[idx, :].values.T  # each row is one node
        xdata = np.asarray(df.loc[idx].iloc[:, 0].index.get_level_values("Time").astype("float"))/time_scale
        # Important to 1) slice only 72 time points with .loc[idx] first, and
        # 2) convert to a numpy array to avoid significant performance bottleneck
        # caused by dataframes.
        # Previous version of the code fortuitously avoided both problems with np.unique
        # in the fitted function, intended to remove duplicate times for regularization
        # but that call is not necessary anymore, so need to do conversion here.

        # Each row contains one node, each column is one time point. Required by curve_fit
        try:
            popt, pcov, jac = curve_fit_jac(func, xdata=xdata, ydata=ydata,
                              absolute_sigma=True, bounds=bounds, p0=p0, reg_rate=reg_rate,
                              offsets=offsets, func_kwargs=func_kwargs)
        except RuntimeError as e:
            print("Could not fit {}".format(idx))
            popt, pcov = np.full([nparams], np.nan), np.full([nparams, nparams], np.nan)
            jac = np.full([ydata.size, nparams], np.nan)

        # Store the results in dataframes
        df_params.loc[idx, param_labels] = popt
        df_params.loc[idx, var_param_labels] = np.diag(pcov)

        # Remove rows in the jacobian associated to regularization
        jac = jac[:ydata.size]  # in total, m dimensions x len(xdata) points = size of ydata matrix
        # Rescale derivatives by the popt value of each parameter, so we have the jacobian
        # as if parameters were rescaled to 1 at the optimum. Equivalent to computing df/d(log(param))
        jac = jac * np.expand_dims(popt, axis=0)
        # Hessian approximately H = J^T J
        hess = np.dot(jac.T, jac)
        df_hess.loc[idx, :] = np.array(hess).reshape(-1)

    return df_params, df_hess


def combine_spline_df_with_fit(df, df_fit):
    """Combine splines and parameterized curves

    Args:
        df (pd.DataFrame): latent space data sampled from splines
        df_fit (pd.DataFrame): data sampled from parameterized curves

    Returns:
        df_compare (pd.DataFrame): combined dataframe with additional index levels (feature, Processing type)
    """

    # Add Processing type index level
    df["Processing type"]="Splines"
    df_fit["Processing type"]="Fit"

    # Combine dataframes
    df_compare=pd.concat([df,df_fit])
    df_compare.set_index("Processing type",inplace=True,append=True)

    # Compute derivatives of df_compare data and return dataframe to same format
    tmp=df_compare[["Node 1","Node 2"]].stack().unstack("Time")
    tmp[0]=0
    tmp=tmp.sort_index(axis=1).astype("float").diff(axis=1).dropna(axis=1).unstack().stack("Time")
    tmp.columns = pd.MultiIndex.from_product([ ['concentration'], tmp.columns])
    tmp=tmp.swaplevel("Processing type","Time")

    # Add feature as column level, combine integral and derivatives, stack dataframe and rename feature index level
    df_compare.columns = pd.MultiIndex.from_product([ ['integral'], df_compare.columns])
    df_compare=pd.concat([df_compare,tmp],axis=1).stack(0)
    df_compare.index.names=df_compare.index.names[:-1]+["Feature"]

    return df_compare


def return_param_and_fitted_latentspace_dfs(df, fittingFunctionName, reg_rate=1.):
    """Returns parameterized dataframes to put into plotting GUI
    Args:
        df (pd.DataFrame): latent space data sampled from splines
        fittingFunctionName (str): name of fitting function; must be a key value in all dictionaries below
        reg_rate (float or None): if None, no regularization. Else, regularization rate in curve_fit_jac
    Returns:
        df_params (pd.DataFrame): contains all parameters from fitting function
        df_compare (pd.DataFrame): combined dataframe with additional index levels (feature, Processing type)
        df_hess (pd.DataFrame): the Hessian matrices of the fit for all conditions.
            The columns give the param i, param j index of the hessian's entry.
            The hessian is obtained as J^T J, where J is the jacobian of the fit,
            scaled by the value of the parameters at the optimum (equivalent
            to computing derivatives with respect to log(param)).
    """
    # Time rescaling defined here along with models and inputted to other functions
    time_scale = 20
    duration = df.index.get_level_values("Time").unique().max()

    # TODO: add a separate option for free alpha.
    param_labels_dict = {
        'Constant velocity':["v0", "t0", "theta", "vt"],
        'Constant force':["F", "t0", "theta", "vt"],
        'Sigmoid':["a0", "t0", "theta", "v1", "gamma"]
    }
    func_dict = {'Constant velocity':ballistic_v0,
                 'Constant force': ballistic_F,
                 'Sigmoid':ballistic_sigmoid}
    bounds_dict = {
        'Constant velocity':[(0, 0, 0, 0), (5, 5, np.pi, 5)],
        'Constant force': [(0, 0, 0, 0), (5, 5, np.pi, 5)],
        'Sigmoid':[(0, 0, -2*np.pi/3, 0, time_scale/50),
                    (5, (duration + 20)/time_scale, np.pi/3, 1, time_scale/2)]
        }
    p0_dict = {
        'Constant velocity':[1, 1, 1, 1],
        'Constant force':[1, 1, 1, 1],
        'Sigmoid':[1, 30 / time_scale, 0, 0.1, 1/7 * time_scale]
    }
    param_offsets_dict = {
        'Constant velocity': np.zeros(4),
        'Constant force': np.zeros(4),
        'Sigmoid': np.array([0, 0, -np.pi/2, 0, 1])
    }

    # Fit curves
    func = func_dict[fittingFunctionName]
    bounds = bounds_dict[fittingFunctionName]
    p0 = p0_dict[fittingFunctionName]
    param_labels = param_labels_dict[fittingFunctionName]
    if reg_rate is None or abs(reg_rate) < 1e-16:
        param_offsets = None  # Avoid a warning in curve_fit_jac
    else:
        param_offsets = param_offsets_dict[fittingFunctionName]


    datasetParamList = []
    datasetFitList = []
    datasetHessList = []  # hessian without the regularization term contribution
    datasets = list(pd.unique(df.index.get_level_values('Data')))
    for dataset in datasets:
        datasetDf = df.xs([dataset], level=['Data'])
        global vt_vm_ratio
        vt_vm_ratio = compute_vt_vm(datasetDf)
        dataset_df_params, dataset_df_hess = return_fit_params(
            datasetDf, func, bounds, p0, param_labels, time_scale=time_scale,
            reg_rate=reg_rate, offsets=param_offsets)

        # Compute latent space coordinates from parameters fits
        dataset_df_fit = datasetDf.copy()
        for idx in dataset_df_params.index:
            fittingVars = dataset_df_params.loc[idx,param_labels]
            times = dataset_df_fit.loc[idx,:].index.get_level_values("Time").astype("float")/time_scale
            if not np.isnan(fittingVars.iloc[0]):
                dataset_df_fit.loc[idx, :] = func(times, *fittingVars).T
            else:
                dataset_df_fit.loc[idx, :] = np.nan

        datasetParamList.append(dataset_df_params)
        datasetFitList.append(dataset_df_fit)
        datasetHessList.append(dataset_df_hess)

    df_params = pd.concat(datasetParamList, keys=datasets,names=['Data'])
    df_fit = pd.concat(datasetFitList, keys=datasets,names=['Data'])
    df_hess = pd.concat(datasetHessList, keys=datasets, names=['Data'])

    # Compare fit vs splines
    df_compare = combine_spline_df_with_fit(df, df_fit)

    return df_params, df_compare, df_hess

### CODE FOR THE GUI
# TODO: bind the Sigmoid model to the GUI as well:
#   - Regularization rate as an option when selecting the model to fit (change collectInputs)
#   - Option to plot the FIM eigenvectors and eigenvalues (add an option in the class PlottingDataframe...)
class FittingFunctionSelectionPage(tk.Frame):
    def __init__(self, master, df, bp):
        tk.Frame.__init__(self, master)
        global backPage
        global dfToParameterize
        dfToParameterize = df.copy()
        print(np.mean(df['Node 1'].values))
        backPage = bp

        mainWindow = tk.Frame(self)
        l1 = tk.Label(mainWindow, text="Select function to fit latent space with:",
            font='Helvetica 18 bold')
        l1.grid(row=0, column=0, sticky=tk.W)
        mainWindow.pack(side=tk.TOP, padx=10, pady=10)

        functionList = ['Constant velocity', 'Constant force', 'Sigmoid']
        functionVar = tk.StringVar(value=functionList[0])
        rblist = []
        for i, function in enumerate(functionList):
            rb = tk.Radiobutton(mainWindow, text=function, padx=20,
                variable=functionVar, value=function)
            rb.grid(row=i+1, column=0, sticky=tk.W)
            rblist.append(rb)

        # Regularization rate as an input. Entry box.
        regrate_memory = tk.StringVar(value="1.")  # Default value, can be converted to float
        l2 = tk.Label(mainWindow,
            text="Enter a regularization rate (positive float):",
            font="Helvetica 18 bold")
        l2.grid(row=len(functionList)+1, column=0, sticky=tk.W)

        # Add a label explaining how to remove focus to validate the value
        l3Var = tk.StringVar(value="Enter a non-negative float")
        l3 = tk.Label(mainWindow, textvariable=l3Var,
            font="Helvetica 12").grid(row=len(functionList)+3,
                column=0, sticky=tk.W)

        # Float input validation
        def validate_float(st):
            try: f = float(st)
            except ValueError:
                check = False
            else:
                if f >= 0.:
                    check = True
                else:
                    check = False
            return check

        # Main validation and correction function
        # Always return True, because we manually revert the change if needed.
        def validate_main(reason, former, proposed):
            # Save the current variable before the user edits
            if reason == "focusin":
                regrate_memory.set(former)
                l3Var.set("Press TAB to validate")
            # Check the proposed string after user edits, rollback if needed
            elif reason == "focusout":
                l3Var.set("Enter a non-negative float")
                if validate_float(proposed):
                    regrate_memory.set(proposed)
                    print("Regularization rate chosen:", proposed)
                else:
                    # Put back the value prior to the user focusin on the box
                    # Won't be called before regrateEntry is created, so no bug
                    regrateEntry.delete(0, tk.END)
                    regrateEntry.insert(0, regrate_memory.get())
            else:
                pass
            return True

        # And registered commands
        fake_widget = tk.Entry()
        validate_cmd = (fake_widget.register(validate_main), "%V", "%s", "%P")
        # Is this necessary since no substitution codes?

        # Now create the Entry with its validation commands
        regrateEntry = tk.Entry(mainWindow, bg="white",
            exportselection=0,
            width=6, # allow 0. to 1., 1.2e-5, etc.
            validate="focus", # Validate after the user has finished typing
            validatecommand=validate_cmd)
            # To validate, output the proposed new text to the validate_float function)
        regrateEntry.grid(row=len(functionList)+2, column=0, sticky=tk.W)
        regrateEntry.insert(0, regrate_memory.get())

        def collectInputs():
            # Get the inputs, function to fit and regularization rate
            functionName = functionVar.get()
            try:
                regrate = float(regrateEntry.get())
            except ValueError:
                regrate = -1  # To enter the condition below
            if regrate < 0.:
                print("The last inputted regularization rate was invalid; ")
                print("Reverting to the last validated rate. ")
                regrate = float(regrate_memory.get())
            print("Regularization rate chosen:", regrate)

            global df_params_plot, df_compare_plot
            # This is the only place where return_param_... is used.
            df_params_plot, df_compare_plot, df_hess_plot = \
                return_param_and_fitted_latentspace_dfs(dfToParameterize, functionName, reg_rate=regrate)
            # TODO: use df_hess_plot for a Fisher info. analysis of the fit.

            df_params_columns = list(df_params_plot.columns)
            df_compare_columns = list(df_compare_plot.columns)

            #Sort dataframes
            sorted_df_params_plot = set_standard_order(df_params_plot.reset_index())
            df_params_plot = pd.DataFrame(sorted_df_params_plot.loc[:,df_params_columns].values,
                index=pd.MultiIndex.from_frame(sorted_df_params_plot.iloc[:,:-1*len(df_params_columns)]))
            df_params_plot.columns = df_params_columns

            sorted_df_compare_plot = set_standard_order(df_compare_plot.reset_index())
            df_compare_plot = pd.DataFrame(sorted_df_compare_plot.loc[:,df_compare_columns].values,
                index=pd.MultiIndex.from_frame(sorted_df_compare_plot.iloc[:,:-1*len(df_compare_columns)]))
            df_compare_plot.columns = df_compare_columns

            print(df_compare_plot)
            print(df_params_plot)

            #Save dataframes:
            projectionName = pickle.load(open(path+'scripts/gui/plotting/projectionName.pkl','rb'))
            with open(path+'output/parameter-dataframes/params-'+projectionName+'-'+functionName+'.pkl','wb') as f:
                pickle.dump(df_params_plot,f)
            with open(path+'output/parameter-space-dataframes/paramSpace-'+projectionName+'-'+functionName+'.pkl','wb') as f:
                pickle.dump(df_compare_plot,f)

            master.switch_frame(PlottingDataframeSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Back",command=lambda: master.switch_frame(backPage)).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)

class PlottingDataframeSelectionPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        mainWindow = tk.Frame(self)
        l1 = tk.Label(
            mainWindow,
            text="Select type of parameterized dataframe to plot:",
            font='Helvetica 18 bold'
            ).grid(row=0,column=0,sticky=tk.W)
        mainWindow.pack(side=tk.TOP,padx=10,pady=10)

        dfTypeList = ['parameters','parameterized latent space']
        dfTypeVar = tk.StringVar(value=dfTypeList[0])
        rblist = []
        for i,dfType in enumerate(dfTypeList):
            rb = tk.Radiobutton(mainWindow, text=dfType,padx = 20, variable=dfTypeVar,value=dfType)
            rb.grid(row=i+1,column=0,sticky=tk.W)
            rblist.append(rb)

        def collectInputs():
            dfType = dfTypeVar.get()
            if dfType == 'parameters':
                df_plot = df_params_plot.copy()
            else:
                df_plot = df_compare_plot.copy()
                df_plot = df_plot.swaplevel(i=-3,j=-2)
                df_plot = df_plot.swaplevel(i=-2,j=-1)
                df_plot_reset = df_plot.reset_index()
                df_plot = pd.DataFrame(df_plot_reset.iloc[:,-3:].values,
                    index=pd.MultiIndex.from_frame(df_plot_reset.iloc[:,:-3]),
                    columns=list(df_plot_reset.columns)[-3:])
                df_plot = df_plot[['Node 1','Node 2','Time']]
                print(df_plot)

            master.switch_frame(selectLevelsPage,df_plot,PlottingDataframeSelectionPage)

        buttonWindow = tk.Frame(self)
        buttonWindow.pack(side=tk.TOP,pady=10)
        tk.Button(buttonWindow, text="OK",command=lambda: collectInputs()).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(
            buttonWindow, text="Back",
            command=lambda: master.switch_frame(FittingFunctionSelectionPage,dfToParameterize,backPage)
            ).pack(in_=buttonWindow,side=tk.LEFT)
        tk.Button(buttonWindow, text="Quit",command=lambda: quit()).pack(in_=buttonWindow,side=tk.LEFT)
