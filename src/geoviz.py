# This to do geographical visualisation

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np 
from cartiflette.download import get_vectorfile_ign
# need topojson to install package

df = pd.read_csv('data/countdeath2020.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)
df['Total'] = df.sum(axis=1)
df_met = pd.DataFrame(df.drop([96,97,98,99,100])['Total'])

dep = get_vectorfile_ign(
  level = "DEPARTEMENT", field = "metropole",
  source = "COG", provider="opendatarchives")
dep["area"] = dep.area

df_met['geometry'] = dep['geometry']
df_geo = gpd.GeoDataFrame(df_met)

ax = df_geo.plot(column ="Total", legend=True)
ax.set_axis_off()
ax.set_title("Mortality COVID-19 in 2020")
plt.show()