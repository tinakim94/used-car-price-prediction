import pandas as pd
import joblib

model = joblib.load("model/car_price_model.pkl")

car = pd.DataFrame([
    {
        "year": 2021,
        "mileage": 30000,
        "engine_size": 2.0,
        "horsepower": 190,
        "brand": "Hyundai",
        "transmission": "Automatic"
    }
])

prediction = model.predict(car)

print(f"예상 중고차 가격: {prediction[0]:,.0f} 원")