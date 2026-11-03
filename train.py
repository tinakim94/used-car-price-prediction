import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. 데이터 불러오기
df = pd.read_csv("data/cars.csv")

print("데이터 확인")
print(df.head())
print("\n데이터 크기:", df.shape)
print("\n컬럼:")
print(df.columns.tolist())


# 2. 필요한 컬럼만 사용
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


# 3. 입력값 / 정답 분리
X = df[features]
y = df[target]


# 4. 컬럼 구분
numeric_features = [
    "year",
    "km_driven"
]

categorical_features = [
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# 5. 전처리
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# 6. RandomForest 모델
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# 7. 전처리 + 모델 Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 8. Train / Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 9. 학습
print("\n모델 학습 시작...")

pipeline.fit(X_train, y_train)

print("모델 학습 완료")


# 10. 예측
predictions = pipeline.predict(X_test)


# 11. 평가
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

print("\n=== 모델 평가 ===")
print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²: {r2:.3f}")


# 12. 모델 저장
os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "model/car_price_model.pkl"
)

print(
    "\n모델 저장 완료:"
    " model/car_price_model.pkl"
)