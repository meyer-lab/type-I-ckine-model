from imports import importData
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

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