import os
from os.path import join
from imports import importData
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
import time
import numpy as np
import pandas as pd

path_here = os.path.dirname(os.path.dirname(__file__))

def getGPData(combCD4=True):
    #import data
    fullData = importData()

    HotEnc = OneHotEncoder()
    HotEnc.fit(fullData[['Cell']])
    HotEncName = HotEnc.transform(fullData[['Cell']]).toarray()
    HotEncName = pd.DataFrame(HotEncName)

    xData = fullData[["Dose","Time","IL2RaKD","IL2RBGKD","IL15Ra","IL2Ra","IL2Rb","IL7Ra","gc","Bivalent"]]
    xData[['Dose', 'IL2RaKD', 'IL2RBGKD']] = np.log10(xData[['Dose', 'IL2RaKD', 'IL2RBGKD']])

    yData = fullData[['Mean']]

    cellNames = ['CD8','NK','Thelper','Treg'] 
    for i, name in enumerate(cellNames): #Adds columns of each cell type w/ hot encoding values
        xData[name]=HotEncName.iloc[:,i]

    if combCD4 == True:
        for i in range(len(xData[['Treg']])):  #For every value in Treg Column
            if xData.at[i,'Treg'] == 1: #If it is equal to 1
                xData.at[i,'Thelper'] = 1  #Set Thelper equal to 1 at this index
        
        xData = xData.rename(columns={'Thelper':'CD4'})
        xData = xData.drop(columns=["Treg"])
    
    return xData, yData, fullData

def gaussianProcess(xData, yData, fData, kernType):

    kern_scale = np.ones(len(xData.columns))

    if kernType == "matern":
        kernel = 1.0 * Matern(length_scale=kern_scale, nu=1.5) 
    elif kernType == "RBF":
        kernel = 1.0 * RBF(kern_scale)
    elif kernType == "combo*":
        kernel = 1.0 * Matern(length_scale=kern_scale, nu=1.5) * 1.0 * RBF(kern_scale)
    elif kernType == "combo+":
        kernel = 1.0 * Matern(length_scale=kern_scale, nu=1.5) + 1.0 * RBF(kern_scale)

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    #use train_test_split to take random selection
    X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.25, random_state=0)

    gp.fit(X_train,y_train)
    return gp

def gaussianTest(xData,yData,testSize, kernType):
    
    kern_scale = np.ones(len(xData.columns))

    if kernType == "matern":
        kernel = 1.0 * Matern(length_scale=kern_scale, nu=1.5) 
    elif kernType == "RBF":
        kernel = 1.0 * RBF(kern_scale)
    elif kernType == "combo*":
        kernel = 1.0 * Matern(length_scale=kern_scale, nu=1.5) * 1.0 * RBF(kern_scale)
    elif kernType == "combo+":
        kernel = 1.0 * Matern(length_scale=kern_scale, nu=1.5) + 1.0 * RBF(kern_scale)

    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    #use train_test_split to take random selection
    X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=testSize, random_state=0)
    
    start_time = time.time()
    gp.fit(X_train,y_train)
    end_time = (time.time() - start_time)
    Rsquare = gp.score(X_test, y_test)

    
    return Rsquare, end_time

xData, yData, fullData = getGPData()
kerns = ["mater","RBF","combo*","combo+"] 
FitData = pd.DataFrame(columns=['Kernel','TestSize','R^2','Time'])
for k in kerns:
    for i in np.linspace(0.1,0.5,9): 
        Rsquare, end_time = gaussianTest(xData,yData,i,k)
        data = {'Kernel':[k],'TestSize':[i],'R^2':[Rsquare],'Time':[end_time]}
        df=pd.DataFrame(data)
        FitData = FitData.append(df, ignore_index=True)

print(FitData)
path = path_here + "/ckine/FitTestData.csv"
FitData.to_csv(str(path), index=False, header=True)
