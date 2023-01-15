"""This module to preprocess the data in order to count the covid deceased"""

import numpy as np
import pandas as pd
import geopandas as gpd

# Preprocessing for the departmental level: 

df2020 = pd.read_csv('data/DC_2020_det.csv', delimiter=";", header = 0)
df2020["DATDEC"] = pd.to_datetime(
                 [str(df2020.ADEC[i]) + "/" + str(df2020.MDEC[i]) + "/" + str(df2020.JDEC[i]) for i in range(df2020.shape[0])],
                 format="%Y/%m/%d"
                )
df2020.drop(["ADEC","MDEC", "COMDEC","JDEC","ANAIS","MNAIS","JNAIS","SEXE","COMDOM","LIEUDEC2"], axis=1, inplace=True)

# Dataframe by county / by day:

dep=pd.unique(df2020["DEPDEC"])
date=pd.unique(df2020["DATDEC"])
df2020["cst"] = [1 for i in range(df2020.shape[0])]
df2020.set_index(["DATDEC","DEPDEC"], inplace = True)

df_dep = pd.DataFrame({i:[int(df2020.loc[(i,j)].count()[0]) if (i,j) in df2020.index else 0 for j in dep] for i in date}, index=dep)

# Now we have a dataframe with all the death by day by county

df_dep.to_csv('data/countdeath2020.csv')

dfspatial = gpd.read_file("./data/contour-des-departements.geojson") 
dfspatial.drop(["nom"])
dfspatial.set_index("code")
dfspatial.to_geojson('data/countdeath2020.csv')

#Importing the data for the covariables: 

dic_demographie = pd.read_excel("./data/covariables_demographiques.xlsx",sheet_name = [2, 3], header = [0,0], skiprows = 2, nrows = 101)

df_covariables = dic_demographie[2].merge(dic_demographie[3], on=(('DEP','DEP'),('DEP','DEP')))
df_covariables.columns = ['Departement','nom_depart1', 'Densite', 'nom_depart2', '25ans']
df_covariables.drop(["nom_depart1",'nom_depart2'], inplace = True, axis = 1)
df_covariables.drop([i if df_covariables.loc[i,"Departement"] in [971, 974, 976] else -1 for i in df_covariables.index], inplace = True, axis = 0, errors='ignore')

df_population = pd.read_excel("./data/pop_2020.xlsx", header = [0,1], skiprows = 3, nrows = 101)
df_population = df_population[[("Départements", 'Unnamed: 0_level_1'),('Ensemble','Total')]]
df_population.columns = ["Departement", "pop"]

df_covariables = df_covariables.merge(df_population, on=("Departement","Departement")).set_index("Departement")
df_covariables = (df_covariables - df_covariables.mean(axis = 0))/df_covariables.std(axis = 0)
df_covariables.to_csv('data/covariables.csv', index = True)

# A df for death from covid only this time during the first wave:

dfcovid = pd.read_csv('data/DC_covid_1erevague.csv', delimiter=";", header = 0, parse_dates = ['jour'])
dfcovid.set_index("jour", inplace = True)
dfcovid = dfcovid.loc["2020-05-17"]
dfcovid = pd.DataFrame({"Mort":dfcovid.groupby("dep")["Dc_Elec_Covid_cum"].sum()})
dfcovid.drop(['971', '972', '973', '974', '975', '976', '977', '978'], inplace = True)

dfcovid.to_csv('data/DC_covid_preprocess.csv')
