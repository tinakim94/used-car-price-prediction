import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


df = pd.read_csv("data/cars.csv")

features = [
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

target = "selling_price"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]


categorical_features = [
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("XGBoost 모델 학습 시작...")

pipeline.fit(
    X_train,
    y_train
)

print("XGBoost 모델 학습 완료")


predictions = pipeline.predict(X_test)


mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n=== XGBoost 모델 평가 ===")

print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²: {r2:.3f}")


joblib.dump(
    pipeline,
    "model/car_price_xgb.pkl"
)

print(
    "\nXGBoost 모델 저장 완료:"
    " model/car_price_xgb.pkl"
)