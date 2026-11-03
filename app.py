import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Used Car Price Prediction API",
    description="XGBoost 기반 중고차 가격 예측 API",
    version="1.0"
)


model = joblib.load(
    "model/car_price_xgb.pkl"
)


class CarInput(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str


@app.get("/")
def root():
    return {
        "message": "Used Car Price Prediction API",
        "status": "running"
    }


@app.post("/predict")
def predict_price(car: CarInput):

    data = pd.DataFrame([
        {
            "year": car.year,
            "km_driven": car.km_driven,
            "fuel": car.fuel,
            "seller_type": car.seller_type,
            "transmission": car.transmission,
            "owner": car.owner
        }
    ])

    prediction = model.predict(data)[0]

    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "INR"
    }