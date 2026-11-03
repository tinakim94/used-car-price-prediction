import sqlite3
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# =========================================================
# 1. SQLite에서 데이터 조회
# =========================================================

conn = sqlite3.connect("data/cars.db")

query = """
SELECT
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    selling_price
FROM cars
WHERE selling_price IS NOT NULL
"""

df = pd.read_sql_query(query, conn)

conn.close()

print("SQL 조회 완료")
print("데이터 크기:", df.shape)


# =========================================================
# 2. Feature / Target 설정
# =========================================================

features = [
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

target = "selling_price"

X = df[features]
y = df[target]


# =========================================================
# 3. 범주형 데이터 전처리
# =========================================================

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


# =========================================================
# 4. XGBoost 파라미터
# =========================================================

params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1
}


# =========================================================
# 5. 모델 생성
# =========================================================

model = XGBRegressor(**params)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================================================
# 6. Train / Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 7. MLflow Experiment 설정
# =========================================================

mlflow.set_experiment(
    "used-car-price-prediction"
)


# =========================================================
# 8. 모델 학습 + MLflow 기록
# =========================================================

with mlflow.start_run():

    print("\nXGBoost 모델 학습 시작...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("XGBoost 모델 학습 완료")


    # -----------------------------------------------------
    # 예측
    # -----------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )


    # -----------------------------------------------------
    # 성능 평가
    # -----------------------------------------------------

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

    print(
        f"MAE: {mae:,.0f}"
    )

    print(
        f"RMSE: {rmse:,.0f}"
    )

    print(
        f"R²: {r2:.3f}"
    )


    # -----------------------------------------------------
    # MLflow Parameter 기록
    # -----------------------------------------------------

    mlflow.log_param(
        "model_type",
        "XGBoost"
    )

    mlflow.log_params(
        params
    )


    # -----------------------------------------------------
    # MLflow Metric 기록
    # -----------------------------------------------------

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    mlflow.log_metric(
        "r2",
        r2
    )


    # -----------------------------------------------------
    # FastAPI에서 사용할 모델 저장
    # -----------------------------------------------------

    joblib.dump(
        pipeline,
        "model/car_price_xgb.pkl"
    )

    print(
        "\n로컬 모델 저장 완료:"
        " model/car_price_xgb.pkl"
    )


    # -----------------------------------------------------
    # MLflow Model Artifact 저장
    #
    # XGBoost 타입을 skops가 차단하는 문제를 피하기 위해
    # cloudpickle 방식 사용
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="model",
        serialization_format="cloudpickle"
    )


    print("MLflow 모델 기록 완료")


print("\n전체 학습 프로세스 완료")