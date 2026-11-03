# Used Car Price Prediction API

실제 중고차 데이터를 활용해 **가격 예측 모델을 학습하고, SQL 기반 데이터 처리부터 MLflow 실험 추적, FastAPI 서빙, Docker 컨테이너화, 외부 배포까지 구현한 End-to-End 머신러닝 프로젝트**입니다.

---

## 🇰🇷 프로젝트 소개

단순히 모델을 학습하는 데서 끝내지 않고, 실제 ML 서비스 흐름을 경험하는 것을 목표로 구성했습니다.

### 주요 구현 내용

- CarDekho 실제 중고차 데이터 **4,340건** 활용
- SQLite 기반 데이터 저장
- SQL을 활용한 학습 데이터 조회
- RandomForest / XGBoost 회귀 모델 비교
- MAE · RMSE · R² 기반 모델 평가
- MLflow 기반 실험 및 모델 추적
- FastAPI 기반 REST API 구현
- Docker 컨테이너화
- Render 외부 배포

### 모델 선정

RandomForest와 XGBoost를 동일한 평가 지표로 비교하고,  
**MAE · RMSE · R² 기준으로 XGBoost를 최종 모델로 선정**했습니다.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| RandomForest | 167,922 INR | 390,256 INR | 0.501 |
| XGBoost | 166,465 INR | 384,544 INR | 0.515 |

현재 모델은 제한된 피처를 사용하고 있어 성능 개선 여지가 있으며, 이 프로젝트에서는 모델 성능 자체뿐 아니라 **데이터 처리 → 학습 → 실험 관리 → API 서빙 → 배포까지의 전체 ML 서비스 흐름 구현**에 중점을 두었습니다.

### 기술 스택

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SQLite` · `SQL` · `MLflow` · `FastAPI` · `Uvicorn` · `Docker` · `Render`

### Live Demo

Swagger UI:

```text
https://used-car-price-prediction-13ch.onrender.com/docs
```

API Endpoint:

```text
POST /predict
```

> Render Free 환경 특성상 장시간 요청이 없을 경우 서비스가 sleep 상태에 들어가 첫 요청이 느릴 수 있습니다.

### 향후 개선 계획

- `name` 컬럼에서 제조사 및 차종 정보 추출
- 추가 Feature Engineering
- Hyperparameter Tuning
- Cross Validation 적용
- MLflow 기반 다중 실험 비교 고도화
- CI/CD 기반 자동 배포
- 모델 및 API 모니터링
- 데이터 전처리 및 학습 파이프라인 자동화

---

# English

## Project Overview

This project implements an end-to-end machine learning workflow for used car price prediction using real-world CarDekho data.

The project covers the full process from data storage and model training to experiment tracking, API serving, containerization, and deployment.

### Key Features

- Real-world used car dataset processing
- SQLite-based data storage
- SQL-based training data retrieval
- RandomForest and XGBoost regression model comparison
- MLflow-based experiment tracking
- FastAPI-based REST API
- Docker containerization
- Render deployment

---

## Dataset

The project uses a public CarDekho used car dataset containing **4,340 records**.

| Column | Description |
|---|---|
| `name` | Car model name |
| `year` | Manufacturing year |
| `selling_price` | Selling price |
| `km_driven` | Kilometers driven |
| `fuel` | Fuel type |
| `seller_type` | Seller category |
| `transmission` | Transmission type |
| `owner` | Ownership history |

Target variable:

```text
selling_price
```

Currency:

```text
INR
```

---

## Data Pipeline

The raw CSV dataset is stored in SQLite and queried using SQL before model training.

```text
CarDekho CSV
      ↓
SQLite Database
      ↓
SQL Query
      ↓
pandas DataFrame
      ↓
Feature Preprocessing
      ↓
Model Training
      ↓
MLflow Experiment Tracking
      ↓
FastAPI
      ↓
Docker
      ↓
Render
```

Example SQL query:

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

---

## Features

The following features are used for model training:

```text
year
km_driven
fuel
seller_type
transmission
owner
```

Categorical features are transformed using `OneHotEncoder`.

---

## Model Training

Two regression algorithms were evaluated:

- RandomForestRegressor
- XGBoost Regressor

Evaluation metrics:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination

---

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| RandomForest | 167,922 INR | 390,256 INR | 0.501 |
| XGBoost | 166,465 INR | 384,544 INR | 0.515 |

XGBoost achieved slightly better performance across all evaluation metrics and was selected as the final model.

The current model performance is limited by the available features, leaving room for further feature engineering and optimization.

---

## MLflow Experiment Tracking

MLflow is used to track model parameters, metrics, and trained model artifacts.

Tracked information includes:

- Model type
- Hyperparameters
- MAE
- RMSE
- R²
- Trained model artifact

Example tracked parameters:

```text
model_type = XGBoost
n_estimators = 500
learning_rate = 0.05
max_depth = 6
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

Final tracked metrics:

```text
MAE  = 166,465
RMSE = 384,544
R²   = 0.515
```

---

## API

The final XGBoost model is served through a REST API using FastAPI.

### Endpoint

```text
POST /predict
```

### Example Request

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

### Example Response

```json
{
  "predicted_price": 524457.75,
  "currency": "INR"
}
```

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the SQLite database:

```bash
python load_db.py
```

Train the XGBoost model:

```bash
python train_xgb.py
```

Run the FastAPI server:

```bash
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## MLflow UI

Run:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

The MLflow UI can be used to inspect:

- Training runs
- Hyperparameters
- Evaluation metrics
- Model artifacts

---

## Docker

Build the Docker image:

```bash
docker build -t used-car-ml .
```

Run the container:

```bash
docker run -p 8000:8000 used-car-ml
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Live Demo

Swagger UI:

```text
https://used-car-price-prediction-13ch.onrender.com/docs
```

API Endpoint:

```text
POST /predict
```

Deployment stack:

```text
Render
Docker
FastAPI
XGBoost
```

---

## Tech Stack

### Machine Learning

- Python
- pandas
- scikit-learn
- XGBoost

### Data

- SQLite
- SQL

### Experiment Tracking

- MLflow

### Backend

- FastAPI
- Uvicorn

### Deployment

- Docker
- Render

---

## Project Structure

```text
used-car-ml/
├── data/
│   ├── cars.csv
│   └── cars.db
├── model/
│   └── car_price_xgb.pkl
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

---

## What I Learned

Through this project, I practiced the complete machine learning service workflow rather than stopping at model training.

Key learning points include:

- Storing and retrieving machine learning data using SQLite and SQL
- Comparing regression models using quantitative evaluation metrics
- Tracking model parameters and metrics with MLflow
- Serving a trained model through a REST API
- Containerizing an ML application with Docker
- Deploying the API to an external environment
- Verifying API responses through Swagger UI

---

## Future Improvements

- Extract manufacturer and model information from the `name` column
- Additional feature engineering
- Hyperparameter tuning
- Cross-validation
- Additional model comparison
- Improved SQL-based preprocessing
- MLflow-based experiment comparison
- CI/CD-based deployment automation
- Model and API monitoring
- Automated data preprocessing and training pipeline

---

## Author

GitHub: `tinakim94`
