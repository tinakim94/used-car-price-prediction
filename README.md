# 🚗 Used Car Price Prediction API

실제 중고차 데이터를 활용하여 가격을 예측하고,  
**데이터 저장 → SQL 조회 → 모델 학습·평가 → 실험 추적 → API 서빙 → 컨테이너화 → 외부 배포**까지 구현한 End-to-End 머신러닝 프로젝트입니다.

---

## 🇰🇷 프로젝트 소개

단순히 모델을 학습하는 데서 끝내지 않고, 실제 머신러닝 서비스가 만들어지는 전체 흐름을 직접 구현하는 것을 목표로 진행했습니다.

### 주요 구현 내용

- CarDekho 실제 중고차 데이터 **4,340건** 활용
- SQLite 기반 데이터 저장
- SQL 기반 학습 데이터 조회
- RandomForest / XGBoost 회귀 모델 학습 및 비교
- MAE · RMSE · R² 기반 모델 평가
- MLflow 기반 실험 파라미터·평가지표·모델 Artifact 추적
- FastAPI 기반 REST API 구현
- Docker 컨테이너화
- Render 외부 배포
- Swagger UI를 통한 실제 예측 API 호출 검증

---

## 📊 모델 비교

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| RandomForest | 167,922 INR | 390,256 INR | 0.501 |
| **XGBoost** | **166,465 INR** | **384,544 INR** | **0.515** |

RandomForest와 XGBoost를 동일한 평가 지표로 비교한 결과,  
**MAE · RMSE · R² 모두에서 상대적으로 우수한 XGBoost를 최종 모델로 선정했습니다.**

현재 모델은 제한된 Feature를 사용하고 있어 추가적인 성능 개선 여지가 있습니다.

이 프로젝트에서는 단순히 높은 정확도를 만드는 것보다  
**데이터 처리부터 모델 서비스 배포까지 이어지는 ML 엔지니어링 흐름을 직접 구현하는 것**에 중점을 두었습니다.

---

## 🛠 Tech Stack

**Machine Learning**  
`Python` · `pandas` · `scikit-learn` · `XGBoost`

**Data**  
`SQLite` · `SQL`

**Experiment Tracking**  
`MLflow`

**Backend / API**  
`FastAPI` · `Uvicorn`

**Deployment**  
`Docker` · `Render`

---

## 🔗 Live Demo

**Swagger UI**

https://used-car-price-prediction-13ch.onrender.com/docs

**API Endpoint**

```text
POST /predict
```

---

## 🔄 Data & ML Pipeline

```text
CarDekho CSV
      │
      ▼
SQLite Database
      │
      ▼
SQL Query
      │
      ▼
pandas DataFrame
      │
      ▼
Feature Preprocessing
      │
      ▼
RandomForest / XGBoost
      │
      ▼
Model Evaluation
(MAE / RMSE / R²)
      │
      ▼
MLflow Experiment Tracking
      │
      ▼
FastAPI
      │
      ▼
Docker
      │
      ▼
Render
```

---

## 🗃 Dataset

CarDekho 공개 중고차 데이터셋을 사용했습니다.

총 **4,340건**의 데이터로 구성되어 있습니다.

| Column | Description |
|---|---|
| `name` | 차량명 |
| `year` | 연식 |
| `selling_price` | 판매 가격 |
| `km_driven` | 주행 거리 |
| `fuel` | 연료 종류 |
| `seller_type` | 판매자 유형 |
| `transmission` | 변속기 유형 |
| `owner` | 소유 이력 |

### Target

```text
selling_price
```

가격 단위는 **INR (Indian Rupee)** 입니다.

---

## 🔎 SQL 기반 학습 데이터 조회

원본 CSV 데이터를 SQLite 데이터베이스에 적재한 뒤, SQL을 이용해 학습에 필요한 컬럼을 조회합니다.

```sql
SELECT
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    selling_price
FROM cars
WHERE selling_price IS NOT NULL;
```

사용 Feature:

```text
year
km_driven
fuel
seller_type
transmission
owner
```

범주형 Feature는 `OneHotEncoder`를 이용해 전처리했습니다.

---

## 🤖 Model Training

두 개의 회귀 모델을 학습하고 비교했습니다.

- `RandomForestRegressor`
- `XGBRegressor`

평가 지표:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

최종적으로 XGBoost 모델을 선택했습니다.

---

## 📈 MLflow Experiment Tracking

MLflow를 이용해 XGBoost 학습 실험을 기록했습니다.

### Tracking 대상

- Model type
- Hyperparameters
- MAE
- RMSE
- R²
- Trained model artifact

### 주요 Parameter

```text
model_type = XGBoost
n_estimators = 500
learning_rate = 0.05
max_depth = 6
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

### Final Metrics

```text
MAE  = 166,465
RMSE = 384,544
R²   = 0.515
```

MLflow UI에서 학습 Run, Parameter, Metric 및 Model Artifact를 확인할 수 있도록 구성했습니다.

---

## ⚡ FastAPI Model Serving

학습한 XGBoost 모델을 FastAPI 기반 REST API로 제공합니다.

### Endpoint

```text
POST /predict
```

### Request Example

```json
{
  "year": 2018,
  "km_driven": 50000,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner"
}
```

### Response Example

```json
{
  "predicted_price": 524457.75,
  "currency": "INR"
}
```

Swagger UI를 통해 외부 환경에서도 `/predict` 요청과 **HTTP 200 OK 응답을 검증했습니다.**

---

## 💻 Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create SQLite Database

```bash
python load_db.py
```

### 3. Train Model

```bash
python train_xgb.py
```

### 4. Run FastAPI

```bash
uvicorn app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 📊 Run MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

MLflow UI에서 다음 정보를 확인할 수 있습니다.

- Training Runs
- Hyperparameters
- Evaluation Metrics
- Model Artifacts

---

## 🐳 Docker

### Build

```bash
docker build -t used-car-ml .
```

### Run

```bash
docker run -p 8000:8000 used-car-ml
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 📁 Project Structure

```text
used-car-ml/
├── data/
│   └── cars.csv
│
├── model/
│   └── car_price_xgb.pkl
│
├── app.py
├── load_db.py
├── predict.py
├── train.py
├── train_xgb.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

`data/cars.db`는 `load_db.py` 실행 시 로컬에서 생성되며 Git 버전 관리에서는 제외합니다.

MLflow의 로컬 실행 데이터(`mlflow.db`, `mlruns/`) 역시 Git 버전 관리에서 제외합니다.

---

## ✅ What I Learned

이 프로젝트를 통해 모델 학습뿐 아니라 실제 ML 서비스를 구성하는 전체 흐름을 경험했습니다.

- 실제 데이터를 활용한 회귀 문제 구성
- RandomForest와 XGBoost 모델 비교
- MAE · RMSE · R²를 활용한 정량적 모델 평가
- SQLite와 SQL 기반 학습 데이터 관리
- MLflow 기반 Experiment Tracking
- 학습 모델의 FastAPI REST API 서빙
- Docker 기반 컨테이너화
- Render 기반 외부 배포
- Swagger UI 기반 실제 API 동작 검증

---

## 🚀 향후 개선 계획

현재 모델과 서비스 구조를 다음 방향으로 확장할 수 있습니다.

- `name` 컬럼에서 제조사 및 차종 정보 추출
- 추가 Feature Engineering
- Hyperparameter Tuning
- Cross Validation 적용
- MLflow 기반 다중 실험 비교
- 추가 알고리즘 성능 비교
- CI/CD 기반 자동 배포
- 모델 및 API 성능 모니터링
- 데이터 전처리 및 학습 파이프라인 자동화

---

# 🇺🇸 English Summary

## Overview

An end-to-end machine learning project for used car price prediction using real-world CarDekho data.

The project covers the workflow from data storage and model training to experiment tracking, REST API serving, containerization, and external deployment.

### Key Features

- Processed 4,340 real-world used car records
- Stored source data in SQLite and retrieved training data using SQL
- Trained and compared RandomForest and XGBoost regression models
- Evaluated models using MAE, RMSE, and R²
- Selected XGBoost as the final model
- Tracked experiments, parameters, metrics, and model artifacts using MLflow
- Served predictions through a FastAPI REST API
- Containerized the application with Docker
- Deployed the API on Render
- Verified the deployed prediction endpoint through Swagger UI

### Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| RandomForest | 167,922 INR | 390,256 INR | 0.501 |
| **XGBoost** | **166,465 INR** | **384,544 INR** | **0.515** |

### Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SQLite` · `SQL` · `MLflow` · `FastAPI` · `Uvicorn` · `Docker` · `Render`

### Live Demo

https://used-car-price-prediction-13ch.onrender.com/docs

---

## 👤 Author

**GitHub:** `tinakim94`
