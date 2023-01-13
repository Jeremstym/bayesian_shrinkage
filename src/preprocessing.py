# this file to preprocess the data in order to count the covid deceased

import numpy as np
import pandas as pd
import geopandas as gpd

df2020 = pd.read_fwf('data/deces-2020.txt')
df2021 = pd.read_fwf('data/deces-2021.txt')
# if doesn't work, try path = '../data/deces-2020.txt'

df2020.drop(['Unnamed: 1'], axis=1, inplace=True)
df2020['mask'] = df2020['Unnamed: 3'].apply(lambda x: isinstance('str', type(x)))
df2020 = df2020[df2020["mask"]==False].drop("mask", axis=1)
df2020.drop('Unnamed: 3', axis=1, inplace=True)
df2020.columns = ['Name', 'Location', 'Date']
df2020.drop("Date", axis=1, inplace=True)

# To handle date issues, unecessary for now
# df2020 = df2020[~df2020["Date"].str.contains("20200000")]
# df2020 = df2020[~df2020["Date"].str.contains("20190700")]
# df2020 = df2020[~df2020["Date"].str.contains("20191200")]
# df2020["Date"].apply(lambda x: pd.to_datetime(x[0:8]))

df2020['Location'] = df2020['Location'].apply(lambda x: x[14:])
df2020 = df2020.groupby(by='Location').count()

df2020.to_csv('data/countdeath2020.csv')

df2021.columns = ['Name', 'Location', 'Date']
df2021['Location'] = df2021['Location'].apply(lambda x: x[14:])

# df2021['mask'] = df2021['Unnamed: 3'].apply(lambda x: isinstance('str', type(x)))
# df2021 = df2021[df2021["mask"]==False].drop("mask", axis=1)
# df2021.drop('Unnamed: 3', axis=1, inplace=True)

df2021.drop("Date", axis=1, inplace=True)
df2021 = df2021.groupby(by='Location').count()
df2021.to_csv('data/countdeath2021.csv')

#Preprocessing for the departmental level : 

df2020 = pd.read_csv('data/DC_2020_det.csv', delimiter=";", header = 0)

df2020["DATDEC"] = pd.to_datetime([str(df2020.ADEC[i]) + "/" + str(df2020.MDEC[i]) + "/" + str(df2020.JDEC[i]) for i in range(df2020.shape[0])], format="%Y/%m/%d")

df2020.drop(["ADEC","MDEC", "COMDEC","JDEC","ANAIS","MNAIS","JNAIS","SEXE","COMDOM","LIEUDEC2"], axis=1, inplace=True)

#Dataframe by county / by day : 
dep=pd.unique(df2020["DEPDEC"])
date=pd.unique(df2020["DATDEC"])
df2020["cst"] = [1 for i in range(df2020.shape[0])]
df2020.set_index(["DATDEC","DEPDEC"], inplace = True)

df_dep = pd.DataFrame({i:[int(df2020.loc[(i,j)].count()[0]) if (i,j) in df2020.index else 0 for j in dep] for i in date}, index = dep)
#Now we have a dataframe with all the death by day by county

df_dep.to_csv('data/countdeath2020.csv')

dfspatial = gpd.read_file("./data/contour-des-departements.geojson") 
dfspatial.drop(["nom"])
dfspatial.set_index("code")

dfspatial.to_geojson('data/countdeath2020.csv')