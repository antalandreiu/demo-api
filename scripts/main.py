import joblib
import pandas as pd
from dateutil.relativedelta import relativedelta
import model
from fastapi import FastAPI
from datetime import datetime



END_TRAIN = datetime(2021, 12, 1)
PREDICTED_YEARS = 60


app = FastAPI()


@app.get("/getForecast/{region}/{province}/{type_of_exercise}/{tourist_residence}")
def forecast(region: str, province: str, type_of_exercise: str, tourist_residence: str):
    path = f"./../data/{region}/models/{province}/{type_of_exercise}/{tourist_residence}"
    #df = pd.read_csv(f"{path}/df.csv")
    with open(f"{path}/arrivals_model.pkl", "rb") as model:
        forecaster = joblib.load(model)
        prediction = forecaster.forecast(PREDICTED_YEARS)

    prediction = prediction.to_dict()
    values = list(prediction.values())
    indexes = list(prediction.keys())
    #indexes = [END_TRAIN + relativedelta(months=x) for x in range(PREDICTED_YEARS)]

    # df.to_dict(orient="records")
    return{"prediction": values, "indexes": indexes}
