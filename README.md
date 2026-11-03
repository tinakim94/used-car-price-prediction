# Used Car Price Prediction API

## 프로젝트 소개

실제 중고차 데이터를 활용하여 가격을 예측하는 머신러닝 프로젝트입니다.

단순 모델 학습에 그치지 않고,

- SQLite 기반 데이터 저장
- SQL 기반 학습 데이터 조회
- RandomForest / XGBoost 모델 비교
- MLflow 기반 실험 추적
- FastAPI 기반 REST API 구현
- Docker 컨테이너화
- Render 외부 배포

까지 머신러닝 서비스의 전체 흐름을 구현했습니다.

### 주요 결과

- 데이터: CarDekho 중고차 데이터 4,340건
- 최종 모델: XGBoost
- MAE: 166,465 INR
- RMSE: 384,544 INR
- R²: 0.515
- FastAPI `/predict` 엔드포인트 구현
- Docker 기반 컨테이너화
- Render 외부 배포 완료
  
### 향후 개선 계획

현재 모델은 제한된 피처를 사용하고 있어 추가적인 성능 개선 여지가 있습니다.

- `name` 컬럼에서 제조사·차종 정보를 추출해 피처 확장
- 하이퍼파라미터 튜닝 및 교차검증 적용
- MLflow를 활용한 실험 비교 및 모델 관리 고도화
- CI/CD 기반 자동 배포 및 모델 업데이트 파이프라인 구성
- API 및 모델 성능 모니터링 추가

### Live Demo

Swagger UI:

https://used-car-price-prediction-13ch.onrender.com/docs

-----

## English

# Used Car Price Prediction API

A machine learning project that predicts used car prices using real-world CarDekho data.

The project covers the full machine learning workflow from data storage and preprocessing to model training, experiment tracking, API serving, containerization, and deployment.

---

## Project Overview

This project was built to practice and implement an end-to-end machine learning service pipeline.

The workflow includes:

- Real-world used car dataset processing
- SQLite-based data storage
- SQL-based training data retrieval
- RandomForest and XGBoost regression model comparison
- MLflow-based experiment tracking
- FastAPI-based model serving
- Docker containerization
- Render deployment

---

## Dataset

The project uses a public CarDekho used car dataset.

The dataset contains **4,340 records** with the following columns:

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

The target variable is:

```text
selling_price
```

Currency: **INR**

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
Machine Learning Model
    ↓
MLflow Experiment Tracking
    ↓
FastAPI
    ↓
Docker
    ↓
Render
```

The training data is retrieved from SQLite using SQL.

Example:

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

The following features are currently used for model training:

```text
year
km_driven
fuel
seller_type
transmission
owner
```

Categorical variables are transformed using `OneHotEncoder`.

---

## Model Training

Two regression algorithms were evaluated:

- RandomForestRegressor
- XGBoost Regressor

The models were evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination

---

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| RandomForest | 167,922 INR | 390,256 INR | 0.501 |
| XGBoost | 166,465 INR | 384,544 INR | 0.515 |

XGBoost achieved slightly better results across the evaluation metrics and was selected as the final model.

The current model performance is still limited by the available features and dataset structure, leaving room for further feature engineering and model optimization.

---

## MLflow Experiment Tracking

MLflow is used to track the XGBoost training experiment.

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

The trained XGBoost model is served through a REST API using FastAPI.

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

The MLflow UI can be used to inspect training runs, parameters, metrics, and model artifacts.

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

Then open:

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

Deployment environment:

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

### MLOps / Experiment Tracking

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

- Storing and retrieving machine learning data with SQLite and SQL
- Comparing multiple regression models using quantitative evaluation metrics
- Tracking model parameters and metrics with MLflow
- Serving a trained machine learning model through a REST API
- Containerizing an ML application with Docker
- Deploying a machine learning API to an external environment
- Verifying deployed API responses through Swagger UI

---

## Future Improvements

Possible improvements include:

- Extracting manufacturer and model information from the `name` column
- Additional feature engineering
- Hyperparameter tuning
- Cross-validation
- Additional model comparison
- SQL-based preprocessing improvements
- MLflow-based experiment comparison
- CI/CD-based deployment automation
- Model and API monitoring

---

## Author

GitHub: `tinakim94`
