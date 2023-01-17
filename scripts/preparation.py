import data_entry as data_e
import model
import pandas as pd
import numpy as np


PATH_MOVEMENTS = "./../data/TUS/datasets/toscana_e_prov_movimento_per_tipo_esercizio_annuale.csv"
PATH_ORIGINS = "./../data/TUS/datasets/toscana_e_prov_paese_di_origine.csv"
PATH_RESIDENTS = "./../data/TUS/datasets/italia_residenti_per_regione_di_origine_annuale.csv"

df_movements = data_e.create_movement_df(PATH_MOVEMENTS, "TUS")
df_origins = data_e.create_movement_df(PATH_ORIGINS, "TUS")
df_residents = data_e.create_movement_df(PATH_RESIDENTS, "TUS")

df_movements.to_csv("./../data/TUS/dataframes/tuscany_turism.csv", index=False)
df_origins.to_csv("./../data/TUS/dataframes/tuscany_origins.csv", index=False)
df_residents.to_csv("./../data/TUS/dataframes/tuscany_residents.csv", index=False)

# --------------------------------------------------------
# ML  MODELS

REGIONS = np.array(["TUS"])
PROVINCES = np.array(["AR", "FI", "GR", "LI", "LU", "MS", "PI", "PO", "PT", "SI", "TUS"], dtype=object)
RESIDENCES = np.array(["EXTR", "IT", "WLD"])
EXERCISES = np.array(["EXTRHOT", "HOT", "TOT"])


model.create_file_structure(regions=REGIONS, provinces=PROVINCES, exercises=EXERCISES, residences=RESIDENCES)




