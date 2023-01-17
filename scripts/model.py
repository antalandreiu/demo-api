from pathlib import Path
import pandas as pd
import numpy as np

from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib



END_TRAIN = datetime(2021, 12, 1)


def _select_data(data) -> pd.DataFrame:
    """ Takes a path and returns a dataframe with data suitable for training"""
    years = np.array([str(i) for i in range(2009, 2022)])
    data = data.drop(index=data[data["period"].isin(years)].index, axis=0)
    data.set_index("period", inplace=True)
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d", errors="raise")
    data.asfreq("MS")
    return data



def _create_models(path: str,df:pd.DataFrame, order:tuple, seasonal_order:tuple):
    data_arrivals = df["arrivals"]
    data_presences = df["presences"]

    train_arrivals = data_arrivals[:END_TRAIN]
    train_presences = data_presences[:END_TRAIN]

    #model_arrivals = SARIMAX(train_arrivals, order=order, seasonal_order=seasonal_order)
    model_presences = SARIMAX(train_presences, order=order, seasonal_order=seasonal_order, trend="t")
    #model_presences_gs = gs_arima(train_presences, p_values=order[0], d_values=order[1], q_values=order[2], P_values=seasonal_order[0], D_values=seasonal_order[1], Q_values=seasonal_order[2], s=seasonal_order[3])
    #model_arrivals_fit = model_arrivals.fit()
    model_presences_fit = model_presences.fit()

    #joblib.dump(model_arrivals_fit, f"{path}/arrivals_model.pkl")
    joblib.dump(model_presences_fit, f"{path}/presences_model.pkl")



def create_file_structure(regions:np.array, provinces:np.array, exercises:np.array, residences:np.array):
    full_data = pd.read_csv("./../data/TUS/dataframes/tuscany_turism.csv")
    for region_name in regions:
        reg_mask = full_data["region"] == region_name
        for province_name in provinces:
            p_mask = full_data["province"] == province_name
            for exercise_name in exercises:
                e_mask = full_data["typeOfExercise"] == exercise_name
                for residence_name in residences:
                    res_mask = full_data["countryOfResidence"] == residence_name

                    path = f"./../data/{region_name}/models/{province_name}/{exercise_name}/{residence_name}"
                    Path(f"{path}").mkdir(parents=True, exist_ok=True)

                    data = full_data[(reg_mask & p_mask & res_mask & e_mask)]

                    data = _select_data(data)
                    data.to_csv(f"{path}/df.csv")
                    #get_param_list(4,2,4,4,2,4)
                    _create_models(path, data, (1, 1, 3), (1, 0, 1, 12))

import pickle
from dateutil.relativedelta import relativedelta

END_TRAIN = datetime(2021, 12, 1)
PREDICTED_YEARS = 60
def forecast(region: str, province: str, type_of_exercise: str, tourist_residence: str):
    path = f"./../data/{region}/models/{province}/{type_of_exercise}/{tourist_residence}"
    #df = pd.read_csv(f"{path}/df.csv")
    with open(f"{path}/arrivals_model.pkl", "rb") as model:
        forecaster = joblib.load(model)
        prediction = forecaster.forecast(PREDICTED_YEARS)

    indexes = [END_TRAIN + relativedelta(months=x) for x in range(PREDICTED_YEARS)]

    # df.to_dict(orient="records")
    return{"path": path, "prediction": prediction, "indexes": indexes}

#forecast("TUS", "AR", "HOT", "IT")