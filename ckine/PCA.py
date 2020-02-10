"""
This file includes various methods for flow cytometry analysis.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
from .flow import count_data, gating


def sampleT(smpl):
    """Output is the T cells data (the protein channels related to T cells)"""
    # Features are the protein channels of interest when analyzing T cells
    features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    # Transform to put on log scale
    tform = smpl.transform("hlog", channels=["BL1-H", "VL1-H", "VL4-H", "BL3-H"])
    # Save the data of each column of the protein channels
    data = tform.data[["BL1-H", "VL1-H", "VL4-H", "BL3-H"]][0:]
    # Save pSTAT5 data
    pstat = tform.data[["RL1-H"]][0:]
    return data, pstat, features


def sampleNK(smpl):
    """Output is the NK cells data (the protein channels related to NK cells)"""
    # For NK, the data consists of different channels so the data var. output will be different
    # Output is data specific to NK cells
    # Features for the NK file of proteins (CD3, CD8, CD56)
    features = ["VL4-H", "RL1-H", "BL1-H"]
    # Transform all proteins (including pSTAT5)
    tform = smpl.transform("hlog", channels=["VL4-H", "RL1-H", "BL1-H"])
    # Assign data of three protein channels AND pSTAT5
    data = tform.data[["VL4-H", "RL1-H", "BL1-H"]][0:]
    pstat = tform.data[["BL2-H"]][0:]
    return data, pstat, features


def fitPCA(data, Tcells=True):
    """
    Fits the PCA model to data, and returns the fit PCA model for future transformations
    (allows for consistent loadings plots between wells)
    """
    if Tcells:
        features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        features = ["VL4-H", "RL1-H", "BL1-H"]
    # Apply PCA to the data set
    # setting values of data of selected features to data frame
    xi = data.loc[:, features].values
    # STANDARDIZE DATA --> very important to do before applying machine learning algorithm
    scaler = preprocessing.StandardScaler()
    xs = scaler.fit_transform(xi)
    xs = np.nan_to_num(xs)
    # setting how many components wanted --> PC1 and PC2
    pca = PCA(n_components=2)
    # apply PCA to standardized data set
    PCAobj = pca.fit(xs)
    # creates the loading array (equation is defintion of loading)
    loading = pca.components_.T
    return PCAobj, loading


def appPCA(data, PCAobj, Tcells=True):
    """Applies the PCA algorithm to the data set"""
    if Tcells:
        features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        features = ["VL4-H", "RL1-H", "BL1-H"]
    # setting values of data of selected features to data frame
    xi = data.loc[:, features].values
    # STANDARDIZE DATA --> very important to do before applying machine learning algorithm
    scaler = preprocessing.StandardScaler()
    xs = scaler.fit_transform(xi)
    xs = np.nan_to_num(xs)
    # transform to the prefit pca object
    xf = PCAobj.transform(xs)
    return xf


def pcaPlt(xf, pstat, ax, Tcells=True):
    """
    Used to plot the score graph.
    Scattered point color gradients are based on range/abundance of pSTAT5 data. Light --> Dark = Less --> More Active
    """
    # PCA
    # Setting x and y values from xf
    x = xf[:, 0]
    y = xf[:, 1]
    # Saving numerical values for pSTAT5 data
    pstat_data = pstat.values
    # Creating a figure for both scatter and mesh plots for PCA
    # This is the scatter plot of the cell clusters colored by pSTAT5 data
    # lighter --> darker = less --> more pSTAT5 present
    # Creating correct dimensions
    pstat = np.squeeze(pstat_data)
    # Creating a data from of x, y, and pSTAT5 in order to graph using seaborn
    combined = np.stack((x, y, pstat)).T
    df = pd.DataFrame(combined, columns=["PC1", "PC2", "pSTAT5"])
    # Creating plot using seaborn. Cool note: virdis is visible for individuals who are colorblind.
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    if Tcells:
        sns.scatterplot(x="PC1", y="PC2", hue="pSTAT5", palette="viridis", data=df, s=5, ax=ax, legend=False, hue_norm=(0, 30000))
        points = ax.scatter(df["PC1"], df["PC2"], c=df["pSTAT5"], s=0, cmap="viridis", vmin=0, vmax=30000)  # set style options
    else:
        sns.scatterplot(x="PC1", y="PC2", hue="pSTAT5", palette="viridis", data=df, s=5, ax=ax, legend=False, hue_norm=(0, 8000))
        points = ax.scatter(df["PC1"], df["PC2"], c=df["pSTAT5"], s=0, cmap="viridis", vmin=0, vmax=8000)  # set style options
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    # add a color bar
    cbar = plt.colorbar(points, ax=ax)
    cbar.set_label('pSTAT Level')


def loadingPlot(loading, ax, Tcells=True):
    """Plot the loading data"""
    # Loading
    # Create graph for loading values
    if Tcells:
        features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    else:
        features = ["VL4-H", "RL1-H", "BL1-H"]

    x_load = loading[:, 0]
    y_load = loading[:, 1]

    # Create figure for the loading plot
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    ax.set_xlabel("PC1", fontsize=15)
    ax.set_ylabel("PC2", fontsize=15)
    ax.scatter(x_load, y_load)
    ax.grid()

    for z, feature in enumerate(features):
        # Please note: not the best logic, but there are three features in NK and four features in T cells
        if Tcells:
            if feature == "BL1-H":
                feature = "Foxp3"
            elif feature == "VL1-H":
                feature = "CD25"
            elif feature == "VL4-H":
                feature = "CD4"
            elif feature == "BL3-H":
                feature = "CD45RA"
        else:
            if feature == "VL4-H":
                feature = "CD3"
            if feature == "RL1-H":
                feature = "CD8"
            if feature == "BL1-H":
                feature = "CD56"
        ax.annotate(str(feature), xy=(x_load[z], y_load[z]), fontsize=8)


def pcaAll(sampleType, Tcells=True):
    """
    Use to plot the score and loading graphs for PCA. Assign protein and pstat5 arrays AND score and loading arrays
    This is all the data for each file.
    Want to use for both T and NK cells? Use it twice!
    sampleType is importF for T or NK
    check == "t" for T cells OR check == "n" for NK cells
    """
    # declare the arrays to store the data
    data_array = []
    pstat_array = []
    xf_array = []
    # create the for loop to file through the data and save to the arrays
    # using the functions created above for a singular file
    if Tcells:
        for i, sample in enumerate(sampleType):
            data, pstat, _ = sampleT(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)

    else:
        for i, sample in enumerate(sampleType):
            data, pstat, _ = sampleNK(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)
    return data_array, pstat_array, xf_array, loading

# ************************PCA by color (gating+PCA)******************************


def sampleTcolor(smpl):
    """Output is the T cells data (the protein channels related to T cells)"""
    # Features are the protein channels of interest when analyzing T cells
    features = ["BL1-H", "VL1-H", "VL4-H", "BL3-H"]
    # Transform to put on log scale
    tform = smpl.transform("hlog", channels=["BL1-H", "VL1-H", "VL4-H", "BL3-H", "RL1-H"])
    # Save the data of each column of the protein channels
    data = tform.data[["BL1-H", "VL1-H", "VL4-H", "BL3-H"]][0:]
    # Save pSTAT5 data
    pstat = tform.data[["RL1-H"]][0:]
    colmat = [] * (len(data) + 1)
    for i in range(len(data)):
        if data.iat[i, 0] > 5.115e+03 and data.iat[i, 0] < 6.258e+03 and data.iat[i, 1] > 3.229e+03 and data.iat[i, 1] < 5.814e+03 and data.iat[i, 2] > 6.512e+03:
            if data.iat[i, 3] > 6300:
                colmat.append('r')  # Treg naive
            else:
                colmat.append('darkorange')  # Treg mem
        elif data.iat[i, 0] > 2.586e+03 and data.iat[i, 0] < 5.115e+03 and data.iat[i, 1] > 3.470e+02 and data.iat[i, 1] < 5.245e+03 and data.iat[i, 2] > 6.512e+03:
            if data.iat[i, 3] > 6300:
                colmat.append('g')  # Thelp naive
            else:
                colmat.append('darkorchid')  # Thelp mem
        else:
            colmat.append('c')
    return data, pstat, features, colmat


def sampleNKcolor(smpl):
    """Output is the NK cells data (the protein channels related to NK cells)"""
    # For NK, the data consists of different channels so the data var. output will be different
    # Output is data specific to NK cells
    # Features for the NK file of proteins (CD3, CD8, CD56)
    features = ["VL4-H", "RL1-H", "BL1-H"]
    # Transform all proteins (including pSTAT5)
    tform = smpl.transform("hlog", channels=["VL4-H", "RL1-H", "BL1-H"])
    # Assign data of three protein channels AND pSTAT5
    data = tform.data[["VL4-H", "RL1-H", "BL1-H"]][0:]
    pstat = tform.data[["BL2-H"]][0:]
    # Create a section for assigning colors to each data point of each cell population --> in this case NK cells
    colmat = [] * (len(data) + 1)

    for i in range(len(data)):
        if data.iat[i, 0] > 5.550e03 and data.iat[i, 0] < 6.468e03 and data.iat[i, 2] > 4.861e03 and data.iat[i, 2] < 5.813e03:
            colmat.append('r')  # nk
        elif data.iat[i, 0] > 6.533e03 and data.iat[i, 0] < 7.34e03 and data.iat[i, 2] > 4.899e03 and data.iat[i, 2] < 5.751e03:
            colmat.append('darkgreen')  # bnk
        elif data.iat[i, 0] > 5.976e03 and data.iat[i, 0] < 7.541e03 and data.iat[i, 1] > 6.825e03 and data.iat[i, 1] < 9.016e03:
            colmat.append('blueviolet')  # cd8+
        elif data.iat[i, 0] > 5.50e03 and data.iat[i, 0] < 6.758e03 and data.iat[i, 2] > 6.021e03 and data.iat[i, 2] < 7.013e03:
            colmat.append('midnightblue')  # nkt
        else:
            colmat.append('c')
    return data, pstat, features, colmat


def pcaPltColor(xf, colormat, ax, Tcells=True):
    """
    Used to plot the score graph.
    Scattered point color gradients are based on range/abundance of pSTAT5 data. Light --> Dark = Less --> More Active
    """
    # PCA
    # Setting x and y values from xf
    x = xf[:, 0]
    y = xf[:, 1]
    # Working with pSTAT5 data --> setting min and max values
    # Creating a figure for both scatter and mesh plots for PCA
    ax.set_xlabel("PC 1", fontsize=15)
    ax.set_ylabel("PC 2", fontsize=15)
    ax.set(xlim=(-5, 5), ylim=(-5, 5))
    # This is the scatter plot of the cell clusters colored by pSTAT5 data
    # lighter --> darker = less --> more pSTAT5 present
    colormat = np.array(colormat)
    if Tcells:
        ax.scatter(x[colormat == "c"], y[colormat == "c"], s=0.5, c="c", label="Other", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "g"], y[colormat == "g"], s=0.5, c="g", label="T Helper Naive", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "darkorchid"], y[colormat == "darkorchid"], s=0.5, c="darkorchid", label="T Helper Memory", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "darkorange"], y[colormat == "darkorange"], s=0.5, c="darkorange", label="T Reg Memory", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "r"], y[colormat == "r"], s=0.5, c="r", label="T Reg Naive", alpha=0.3, edgecolors='none')
        ax.legend(markerscale=6.)
    else:
        ax.scatter(x[colormat == "darkgreen"], y[colormat == "darkgreen"], s=0.5, c="g", label="BNK", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "r"], y[colormat == "r"], s=0.5, c="r", label="NK", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "blueviolet"], y[colormat == "blueviolet"], s=0.5, c="blueviolet", label="CD8+", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "midnightblue"], y[colormat == "midnightblue"], s=0.5, c="midnightblue", label="NKT", alpha=0.3, edgecolors='none')
        ax.scatter(x[colormat == "c"], y[colormat == "c"], s=0.5, c="c", label="Other", alpha=0.3, edgecolors='none')
        ax.legend(markerscale=6.)


def pcaAllCellType(sampleType, Tcells=True):
    """
    Use to plot the score and loading graphs for PCA. Assign protein and pstat5 arrays AND score and loading arrays
    This is all the data for each file.
    Want to use for both T and NK cells? Use it twice!
    sampleType is importF for T or NK
    check == "t" for T cells OR check == "n" for NK cells
    """
    # declare the arrays to store the data
    data_array = []
    pstat_array = []
    xf_array = []
    colormat_array = []

    # create the for loop to file through the datfa and save to the arrays
    # using the functions created above for a singular file
    if Tcells:
        for i, sample in enumerate(sampleType):
            data, pstat, _, colormat = sampleTcolor(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)
            colormat_array.append(colormat)
    else:
        for i, sample in enumerate(sampleType):
            data, pstat, _, colormat = sampleNKcolor(sample)
            data_array.append(data)
            pstat_array.append(pstat)
            if i == 0:
                PCAobj, loading = fitPCA(data, Tcells)
            xf = appPCA(data, PCAobj, Tcells)
            xf_array.append(xf)
            colormat_array.append(colormat)
    return data_array, pstat_array, xf_array, loading, colormat_array

# ************************Dose Response by PCA******************************


def PCADoseResponse(sampleType, PC1Bnds, PC2Bnds, cell_type, Tcells=True):
    """
    Given data from a time Point and two PC bounds, the dose response curve will be calculated and graphed
    (needs folder with FCS from one time point)
    """
    dosemat = np.array([84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474])
    pSTATvals = np.zeros([1, dosemat.size])
    if cell_type:
        gates = gating(cell_type)
        _, alldata = count_data(sampleType, gates, Tcells)

    for i, sample in enumerate(sampleType):
        if Tcells:
            data, pstat, _ = sampleT(sample)  # retrieve data
            statcol = 'RL1-H'
        else:
            data, pstat, _ = sampleNK(sample)
            statcol = 'BL2-H'
        if cell_type:
            data = alldata[i]
            pstat = data[[statcol]]

        if i == 0:
            PCAobj, loading = fitPCA(data, Tcells)  # only fit to first set

        xf = appPCA(data, PCAobj, Tcells)  # get PC1/2 vals
        PCApd = PCdatTransform(xf, pstat)
        PCApd = PCApd[(PCApd['PC1'] >= PC1Bnds[0]) & (PCApd['PC1'] <= PC1Bnds[1]) & (PCApd['PC2'] >= PC2Bnds[0]) & (PCApd['PC2'] <= PC2Bnds[1])]  # remove data that that is not within given PC bounds
        pSTATvals[0, i] = PCApd.loc[:, "pSTAT"].mean()  # take average Pstat activity of data fitting criteria

    pSTATvals = pSTATvals.flatten()

    return pSTATvals, dosemat, loading


def PCdatTransform(xf, pstat):
    """Takes PCA Data and transforms it into a pandas dataframe (variable saver)"""
    PC1, PC2, pstat = np.transpose(xf[:, 0]), np.transpose(xf[:, 1]), pstat.to_numpy()
    PC1, PC2 = np.reshape(PC1, (PC1.size, 1)), np.reshape(PC2, (PC2.size, 1))
    PCAstat = np.concatenate((PC1, PC2, pstat), axis=1)
    PCApd = pd.DataFrame({'PC1': PCAstat[:, 0], 'PC2': PCAstat[:, 1], 'pSTAT': PCAstat[:, 2]})  # arrange into pandas datafrome
    return PCApd


def StatGini(sampleType, ax, cell_type, Tcells=True):
    """
    Define the Gini Coefficient of Pstat Vals Across a timepoint for either whole or gated population.
    Takes a folder of samples, a timepoint (string), a boolean check for cell type and an optional gate parameter.
    """
    alldata = []
    dosemat = np.array([[84, 28, 9.333333, 3.111, 1.037037, 0.345679, 0.115226, 0.038409, 0.012803, 0.004268, 0.001423, 0.000474]])
    ginis = np.zeros([2, dosemat.size])

    if Tcells:
        statcol = 'RL1-H'
    else:
        statcol = 'BL2-H'

    if cell_type:
        gates = gating(cell_type)
        _, alldata = count_data(sampleType, gates, Tcells)  # returns array of dfs in case of gate or no gate

    else:
        for i, sample in enumerate(sampleType):
            if Tcells:
                _, pstat, _ = sampleT(sample)
                alldata.append(pstat)
            else:
                _, pstat, _ = sampleNK(sample)
                alldata.append(pstat)

    for i, sample in enumerate(sampleType):  # get pstat data and put it into list form
        dat_array = alldata[i]
        stat_array = dat_array[[statcol]]
        stat_array = stat_array.to_numpy()
        stat_array = stat_array.clip(min=1)  # remove small percentage of negative pstat values
        stat_array.tolist()  # manipulate data to be compatible with gin calculation
        stat_sort = np.sort(np.hstack(stat_array))
        num = stat_array.size
        subconst = (num + 1) / num
        coef = 2 / num
        summed = sum([(j + 1) * stat for j, stat in enumerate(stat_sort)])
        ginis[0, i] = (coef * summed / (stat_sort.sum()) - subconst)\

    for i, sample in enumerate(sampleType):  # Get inverse Ginis
        dat_array = alldata[i]
        stat_array = dat_array[[statcol]]
        stat_array = stat_array.to_numpy()
        stat_array = stat_array.clip(min=1)
        stat_array = np.reciprocal(stat_array)
        stat_array.tolist()
        stat_sort = np.sort(np.hstack(stat_array))
        num = stat_array.size
        subconst = (num + 1) / num
        coef = 2 / num
        summed = sum([(j + 1) * stat for j, stat in enumerate(stat_sort)])
        ginis[1, i] = (coef * summed / (stat_sort.sum()) - subconst)

    ax.plot(dosemat, np.expand_dims(ginis[0, :], axis=0), ".--", color="navy", label="Gini Coefficients")
    ax.plot(dosemat, np.expand_dims(ginis[1, :], axis=0), ".--", color="darkorange", label="Inverse Gini Coefficients")
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel("Cytokine Dosage (log10[nM])")
    ax.set_ylabel("Gini Coefficient")
    ax.set(xlim=(0.0001, 100))
    ax.set(ylim=(0., 1))

    return ginis, dosemat


def nllsq_EC50(x0, xdata, ydata):
    """
    Performs nonlinear least squares on activity measurements to determine parameters of Hill equation and outputs EC50.
    """
    lsq_res = least_squares(residuals, x0, args=(xdata, ydata), bounds=([0., 0., 0., 0.], [10., 100., 10**5., 10**5]), jac='3-point')
    return lsq_res.x[0]


def residuals(x0, x, y):
    """ Residual function for Hill Equation. """
    return hill_equation(x, x0) - y


def hill_equation(x, x0, solution=0):
    """ Calculates EC50 from Hill Equation. """
    k = x0[0]
    n = x0[1]
    A = x0[2]
    floor = x0[3]
    xk = np.power(x / k, n)
    return (A * xk / (1.0 + xk)) - solution + floor


def EC50_PC_Scan(sampleType, min_max_pts, ax, cell_type, Tcells=True, PC1=True):
    """Scans along one Principal component and returns EC50 for slices along that Axis"""
    x0 = [1, 2., 5000., 3000.]  # would put gating here
    EC50s = np.zeros([1, min_max_pts[2]])
    scanspace = np.linspace(min_max_pts[0], min_max_pts[1], num=min_max_pts[2] + 1)
    axrange = np.array([-100, 100])

    for i in range(0, min_max_pts[2]):  # set bounds and calculate EC50s
        if PC1:
            PC1Bnds, PC2Bnds = np.array([scanspace[i], scanspace[i + 1]]), axrange
        else:
            PC2Bnds, PC1Bnds = np.array([scanspace[i], scanspace[i + 1]]), axrange
        pSTATs, doses, loading = PCADoseResponse(sampleType, PC1Bnds, PC2Bnds, cell_type, Tcells)
        doses = np.log10(doses.astype(np.float) * 1e4)
        EC50s[0, i] = nllsq_EC50(x0, doses, pSTATs)

    EC50s = EC50s.flatten() - 4  # account for 10^4 multiplication
    ax.plot(scanspace[:-1] + (min_max_pts[1] - min_max_pts[0]) / (2 * min_max_pts[2]), EC50s, ".--", color="navy")
    ax.grid()

    if PC1:
        ax.set_xlabel("PC1 Space")
    else:
        ax.set_xlabel("PC2 Space")
    ax.set_ylabel("log[EC50] (nM)")
    ax.set(ylim=(-3, 3))
    ax.set(xlim=(min_max_pts[0], min_max_pts[1]))
    ax.grid()

    return loading
