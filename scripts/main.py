import joblib
from fastapi import FastAPI
from datetime import datetime



END_TRAIN = datetime(2021, 12, 1)
PREDICTED_YEARS = 60


app = FastAPI()


@app.get("/{region}/{province}/{type_of_exercise}/{tourist_residence}")
def forecast(region: str, province: str, type_of_exercise: str, tourist_residence: str):

    path = f"./../data/{region}/models/{province}/{type_of_exercise}/{tourist_residence}"
    with open(f"{path}/arrivals_model.pkl", "rb") as model:
        forecaster = joblib.load(model)
        prediction = forecaster.forecast(PREDICTED_YEARS)

    values = list(prediction.array)
    indexes = list(prediction.index)
    assert len(values) == len(indexes)

    data = [{"date": indexes[i],  "npeople": values[i]} for i in range(len(values))]

    return {"prediction": data}

if __name__ == '__main__':
