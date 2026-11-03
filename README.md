## Live Demo

Swagger UI:

https://used-car-price-prediction-13ch.onrender.com/docs

API Endpoint:

POST `/predict`

배포 환경:
- Render
- Docker
- FastAPI
- XGBoost

# Used Car Price Prediction API

중고차 데이터를 활용해 차량 가격을 예측하는 머신러닝 기반 REST API 프로젝트입니다.

RandomForest와 XGBoost 회귀 모델을 동일한 데이터셋과 Train/Test 조건에서 비교하고, 성능이 더 우수한 XGBoost 모델을 최종 모델로 선정했습니다.

최종 모델은 FastAPI 기반 REST API로 제공하며 Docker 컨테이너 환경에서 실행할 수 있도록 구성했습니다.

## Project Goals

- 실제 중고차 데이터를 활용한 회귀 모델 학습
- RandomForest와 XGBoost 모델 성능 비교
- 머신러닝 모델 저장 및 재사용
- FastAPI 기반 Prediction API 구현
- Docker 기반 컨테이너화
- 외부 환경에 배포 가능한 ML 서비스 구성

## Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Docker
- Git / GitHub

## Dataset

CarDekho 중고차 공개 데이터셋을 활용했습니다.

주요 Feature:

- year
- km_driven
- fuel
- seller_type
- transmission
- owner

Target:

- selling_price

## Machine Learning Pipeline

```text
Raw Data
   ↓
Data Preprocessing
   ↓
Train / Test Split
   ↓
RandomForest / XGBoost Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Serialization
   ↓
FastAPI Prediction API
   ↓
Docker Container
   ↓
Deployment
```

## Model Comparison

### RandomForest

- MAE: 167,922
- RMSE: 390,256
- R²: 0.501

### XGBoost

- MAE: 166,465
- RMSE: 388,544
- R²: 0.515

XGBoost가 MAE와 RMSE에서 더 낮은 오차를 보였고 R² 또한 더 높아 최종 모델로 선정했습니다.

## API

### POST `/predict`

Request:

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

Response example:

```json
{
  "predicted_price": 524457.75,
  "currency": "INR"
}
```

## Run Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start FastAPI

```bash
uvicorn app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

Build:

```bash
docker build -t used-car-ml .
```

Run:

```bash
docker run -p 8000:8000 used-car-ml
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## What I Learned

이 프로젝트를 통해 단순히 머신러닝 모델을 학습하는 데 그치지 않고 실제 서비스 형태로 연결하는 전체 흐름을 경험했습니다.

특히 RandomForest를 baseline 모델로 구성한 뒤 XGBoost와 동일한 조건에서 성능을 비교하고 최종 모델을 선정했습니다.

또한 학습한 모델을 파일로 저장하고 FastAPI에서 로드해 REST API 형태로 제공했으며, Docker 컨테이너 환경에서도 동일하게 실행되는 것을 검증했습니다.

이를 통해 데이터 전처리, 모델 학습 및 평가, 모델 서빙, API 구현, 컨테이너화까지 머신러닝 서비스의 전체 흐름을 경험했습니다.

## Future Improvements

- 차량 제조사 및 모델명 Feature Engineering
- Hyperparameter Tuning
- Cross Validation
- 모델 성능 개선 및 추가 알고리즘 비교
- API 요청 Validation 강화
- 모델 및 API 모니터링
- CI/CD 기반 자동 배포
