# Kaggle Playground S6E1 — Predicting Student Test Scores 코드 스터디 노트

> **대회**: Kaggle Playground Series S6E1
> **목표**: 학생 시험 점수(`exam_score`) 예측
> **평가 지표**: RMSE (Root Mean Squared Error, 낮을수록 좋음)
> **최종 전략**: Ridge Meta-Feature + Optimized XGBoost

---

## 목차

1. [전체 파이프라인 요약](#1-전체-파이프라인-요약)
2. [데이터셋 개요](#2-데이터셋-개요)
3. [셀별 코드 상세 분석](#3-셀별-코드-상세-분석)
4. [핵심 개념 딥다이브](#4-핵심-개념-딥다이브)
5. [피처 엔지니어링 총정리](#5-피처-엔지니어링-총정리)
6. [모델링 전략 상세](#6-모델링-전략-상세)
7. [하이퍼파라미터 해부](#7-하이퍼파라미터-해부)
8. [버전별 발전 과정](#8-버전별-발전-과정)
9. [자기 점검 질문 15선](#9-자기-점검-질문-15선)
10. [추가 학습 방향](#10-추가-학습-방향)

---

## 1. 전체 파이프라인 요약

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  데이터 로드  │ →  │  품질 점검 & EDA   │ →  │  피처 엔지니어링     │
│  (train/test) │    │  (누수 방지 체크)   │    │  (11 → 46 features) │
└─────────────┘    └──────────────────┘    └─────────┬───────────┘
                                                     │
                              ┌───────────────────────┘
                              ▼
                   ┌──────────────────────┐
                   │  Stage 1: Ridge CV    │
                   │  (10-Fold + TargetEnc) │
                   │  → OOF 메타피처 생성   │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Stage 2: XGBoost     │
                   │  (46 features + Ridge │
                   │   meta-feature = 47)  │
                   │  → 최종 예측 생성      │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  제출 파일 생성        │
                   │  (submission.csv)     │
                   └──────────────────────┘
```

### 핵심 아이디어

이 노트북의 핵심은 **2-Stage Stacking (Meta-Feature)** 전략입니다.

| 단계 | 모델 | 역할 | 인코딩 방식 |
|------|------|------|-------------|
| Stage 1 | **RidgeCV** | 메타피처 생성자 | TargetEncoder |
| Stage 2 | **XGBoost** | 최종 예측기 | Category type (native) |

> **왜 이 구조인가?**
> - Ridge는 선형 모델이므로 "전체 평균적 경향"을 안정적으로 포착
> - XGBoost는 비선형 패턴/상호작용을 잘 잡음
> - Ridge의 예측값을 피처로 넣으면, XGBoost가 "선형 모델이 보는 세계"를 추가 정보로 활용 가능
> - 단순 블렌딩(가중 평균)보다 더 유연한 조합이 가능

---

## 2. 데이터셋 개요

### 2.1 데이터 규모

| 구분 | 행 수 | 컬럼 수 | 비고 |
|------|-------|---------|------|
| Train | 630,000 | 13 | id + 11 features + exam_score |
| Test | 270,000 | 12 | id + 11 features (타겟 없음) |
| Original | — | — | Kaggle에서만 사용 가능 (augmentation용) |

### 2.2 피처 목록

**수치형 (Numeric) — 4개**

| 피처 | 설명 | 비고 |
|------|------|------|
| `age` | 나이 | 연속형 |
| `study_hours` | 공부 시간 | 핵심 피처 |
| `class_attendance` | 수업 출석률 | 핵심 피처 |
| `sleep_hours` | 수면 시간 | 생활 습관 |

**범주형 (Categorical) — 7개**

| 피처 | 설명 | 유형 |
|------|------|------|
| `gender` | 성별 | 명목형 |
| `course` | 수강 과목 | 명목형 |
| `internet_access` | 인터넷 접근성 | 이진형 |
| `sleep_quality` | 수면 품질 | 순서형 (poor/average/good) |
| `study_method` | 학습 방법 | 명목형 |
| `facility_rating` | 시설 평가 | 순서형 (low/medium/high) |
| `exam_difficulty` | 시험 난이도 | 순서형 (easy/moderate/hard) |

### 2.3 데이터 품질

- **결측값**: 0개 (train/test 모두)
- **중복 행**: 0개
- **타겟 누수**: 없음 (test에 `exam_score` 미포함 확인)

---

## 3. 셀별 코드 상세 분석

### Cell 0 — 문제 정의 (Markdown)

```
타겟: exam_score
지표: RMSE
피처: 46개 엔지니어링 피처 + Ridge 메타피처
모델: Ridge(메타피처) + XGBoost(최종)
CV: 10-Fold + 원본 데이터 증강
```

> **체크포인트**: 노트북 시작 시 전략을 명확히 선언하면, 코드가 길어져도 방향을 잃지 않습니다.

---

### Cell 1 — 환경 세팅 & 경로 탐색

#### 핵심 라이브러리

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import RidgeCV
import xgboost as xgb
```

#### 재현성 확보

```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

> **왜 중요한가?**
> - CV split, 모델의 랜덤 요소가 고정되어 실험 비교가 가능
> - 같은 코드를 돌리면 항상 같은 결과 → 디버깅/개선 추적이 쉬움

#### `find_data_paths()` — 경로 자동 탐색

```python
def find_data_paths():
    # 1순위: Kaggle 환경 경로
    kaggle_train = Path('/kaggle/input/playground-series-s6e1/train.csv')

    # 2순위: 로컬 환경 경로 (여러 후보 탐색)
    data_dir = Path('Predicting Student Test Scores/data')
    ...
```

> **실전 팁**: 데이터 분석 노트북은 모델보다 **경로 문제**로 실패하는 경우가 많습니다.
> 환경(Kaggle/로컬/IDE)이 바뀌어도 자동으로 데이터를 찾는 패턴이 매우 유용합니다.

**확인 사항**:
- [ ] `Train exists: True` 출력 확인
- [ ] `Test exists: True` 출력 확인

---

### Cell 2 — 데이터 로드

```python
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

# 원본 데이터 (Kaggle에서만 사용 가능)
if ORIGINAL_PATH and ORIGINAL_PATH.exists():
    original_df = pd.read_csv(ORIGINAL_PATH)
```

**확인 사항**:
- [ ] Train shape: `(630000, 13)` — 13 = id(1) + features(11) + target(1)
- [ ] Test shape: `(270000, 12)` — 12 = id(1) + features(11), 타겟 없음
- [ ] `CATS` 리스트가 7개 범주형과 일치하는지

> **Original Data란?**
> Kaggle Playground Series는 실제 데이터셋을 기반으로 합성 데이터를 생성합니다.
> 원본 데이터(`Exam_Score_Prediction.csv`)를 학습에 추가하면 모델의 일반화 성능이 향상될 수 있습니다.

---

### Cell 3 — 데이터 품질 점검

```python
# 누수 방지 assert
assert TARGET in train_df.columns        # train에 타겟 존재 확인
assert TARGET not in test_df.columns     # test에 타겟 없음 확인 (누수 방지!)

# 결측값, 중복 체크
train_df.isna().sum().sum()   # → 0
test_df.isna().sum().sum()    # → 0
train_df.duplicated().sum()   # → 0
```

> **왜 `assert`가 중요한가?**
> 실수로 전처리 과정에서 test에 타겟이 포함되면 **데이터 누수(data leakage)** 가 발생합니다.
> 모델이 답을 이미 알고 있으므로 CV 점수는 높지만 실제 제출 점수는 낮아지는 치명적 실수입니다.

---

### Cell 4 — 피처 엔지니어링 (11 → 45개)

이 셀이 **이 노트북의 성능 핵심**입니다. 11개 원본 피처에서 34개의 파생 피처를 추가로 생성합니다.

```python
def preprocess_optimized(df):
    """46개 고가치 피처 생성: 다항식, 로그, 상호작용, 구간화 등"""
```

#### 피처 카테고리별 정리

| 카테고리 | 개수 | 예시 | 목적 |
|---------|------|------|------|
| 다항식 (제곱) | 4 | `study_hours²`, `attendance²` | 비선형 효과 포착 |
| 로그 변환 | 3 | `log(study_hours+1)` | 분포 정규화, 이상치 완화 |
| 제곱근 변환 | 2 | `√study_hours` | 완만한 비선형 변환 |
| 수치 × 수치 상호작용 | 4 | `study_hours × attendance` | 변수 간 시너지 효과 |
| 비율 (Ratio) | 3 | `study_hours / sleep_hours` | 상대적 비교 |
| 순서형 인코딩 | 3 | `sleep_quality → 0/1/2` | 범주형의 순서 정보 활용 |
| 순서형 × 수치 상호작용 | 3 | `study_hours × sleep_quality` | 범주-수치 시너지 |
| 순서형 × 순서형 | 2 | `facility × sleep_quality` | 범주 간 조합 효과 |
| 규칙 기반 플래그 | 3 | `출석≥90 & 공부≥6` | 도메인 지식 반영 |
| 복합 효율 지표 | 1 | `(study × attendance) / (sleep+1)` | 종합 학습 효율 |
| 갭 피처 | 2 | `\|sleep - 8\|` | 이상적 값과의 거리 |
| 구간화 (Binning) | 4 | `study_hours를 5구간으로` | 비선형 경계 포착 |

> **왜 이렇게 많은 피처를 만드나?**
> - 트리 모델(XGBoost)은 피처가 많아도 내부적으로 중요한 것만 선택하므로, 많이 만들어 놓는 것이 유리
> - 다만 선형 모델에서는 다중공선성 문제가 생길 수 있어, Ridge(L2 정규화)로 이를 제어

#### 코드 패턴 상세

**1) 안전한 로그/제곱근 변환**

```python
sh_pos = df_temp['study_hours'].clip(lower=0)  # 음수 방지
df_temp['log_study_hours'] = np.log1p(sh_pos)   # log(x+1)로 0일 때도 안전
```

> `np.log1p(x)` = `np.log(x + 1)` → x=0이어도 `log(1) = 0`으로 안전

**2) 안전한 비율 계산**

```python
eps = 1e-5
df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
```

> `eps`(아주 작은 값)를 더해서 분모가 0이 되는 것을 방지

**3) 순서형 범주 인코딩**

```python
sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map)
```

> 순서가 있는 범주형 변수는 숫자로 변환하면 "크기 관계"를 모델이 학습할 수 있습니다.

**4) 도메인 지식 기반 플래그**

```python
df_temp['high_att_high_study'] = (
    (df_temp['class_attendance'] >= 90) & (df_temp['study_hours'] >= 6)
).astype(int)
df_temp['ideal_sleep_flag'] = (
    (df_temp['sleep_hours'] >= 7) & (df_temp['sleep_hours'] <= 9)
).astype(int)
```

> 교육학 도메인 지식: 출석 90%↑ & 공부 6시간↑ 또는 수면 7~9시간이면 성적이 좋을 가능성 높음

**5) 구간화 (Binning)**

```python
df_temp['study_bin_num'] = pd.cut(df_temp['study_hours'], bins=5, labels=False)
```

> 연속형 변수를 구간으로 나누면 트리 모델이 비선형 경계를 더 잘 찾을 수 있습니다.

---

### Cell 5 — Ridge 메타피처 생성 (Stage 1)

이 셀은 **스태킹(Stacking)의 첫 번째 레이어**를 구현합니다.

```python
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])           # train OOF 저장
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))  # test 예측 (fold별)
```

#### 처리 흐름 (매 Fold마다)

```
1. Train/Valid 분할
      ↓
2. 원본 데이터 결합 (augmentation)
      ↓
3. TargetEncoder로 범주형 인코딩
      ↓
4. RidgeCV 학습 (alpha 자동 선택)
      ↓
5. Valid 예측 → oof_pred_lr[val_index]에 저장
6. Test 예측 → test_preds_lr[:, fold]에 저장
```

#### TargetEncoder 상세

```python
target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS], y_train_combined)
X_val_encoded[CATS]   = target_encoder.transform(X_val[CATS])
```

> **TargetEncoder란?**
> 범주형 변수의 각 카테고리를 **해당 카테고리의 타겟 평균값**으로 치환하는 인코딩 방식입니다.
>
> 예: `gender='Male'`인 학생들의 평균 시험 점수가 65점이면, `Male` → `65`로 변환
>
> **장점**: 카디널리티가 높아도 1차원으로 표현, 타겟과의 관계를 직접 반영
> **주의**: 반드시 fold별로 fit/transform을 분리해야 누수 방지 (`smooth='auto'`로 과적합도 완화)

#### RidgeCV (자동 alpha 선택)

```python
alphas = np.logspace(-3, 3, 20)  # 0.001 ~ 1000 사이 20개
lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
```

> **RidgeCV**는 내부적으로 5-fold CV를 돌면서 최적의 정규화 강도(`alpha`)를 자동으로 찾습니다.
> alpha가 클수록 정규화가 강해지고(계수를 더 작게), 작을수록 일반 선형 회귀에 가까워집니다.

#### 예측값 클리핑

```python
lr_val_pred = np.clip(lr_model.predict(X_val_encoded), 0, 100)
```

> 시험 점수는 0~100 사이이므로, 이 범위를 벗어나는 예측을 잘라냅니다.

#### OOF (Out-of-Fold) 예측의 핵심 원리

```
     Fold 1        Fold 2        Fold 3       ...     Fold 10
   ┌─────────┐  ┌─────────┐  ┌─────────┐          ┌─────────┐
   │ Train    │  │ Train    │  │ Train    │          │ Train    │
   │          │  │          │  │          │          │          │
   │──────────│  │──────────│  │──────────│          │──────────│
   │ Valid ■  │  │ Valid  ■ │  │ Valid  ■ │   ...    │ Valid  ■ │
   └─────────┘  └─────────┘  └─────────┘          └─────────┘
        │              │             │                    │
        ▼              ▼             ▼                    ▼
   oof[idx_1]    oof[idx_2]   oof[idx_3]   ...    oof[idx_10]

   → 모든 train 행에 대해 "자신이 검증셋일 때의 예측값"이 채워짐
```

> **OOF가 왜 중요한가?**
> - train 전체에 대해 "누수 없는" 예측값을 얻을 수 있음
> - 이 예측값을 다음 스테이지의 피처로 사용하면 정보 누수 없이 스태킹 가능

---

### Cell 6 — XGBoost 데이터셋 준비 (Stage 2 입력 구성)

```python
# 범주형을 XGBoost의 native category 타입으로 변환
for col in CATS:
    X_xgb_raw[col] = X_xgb_raw[col].astype(str).astype('category')

# Ridge 메타피처 추가
X_xgb['feature_lr_pred'] = oof_pred_lr                    # train: OOF 예측값
X_test_xgb['feature_lr_pred'] = test_preds_lr.mean(axis=1) # test: fold별 평균
```

#### 왜 XGBoost에서 category 타입을 사용하는가?

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| OneHotEncoder | 각 카테고리를 0/1 열로 변환 | 정보 손실 없음 | 차원 폭발 가능 |
| OrdinalEncoder | 각 카테고리를 정수로 변환 | 빠름 | 순서 정보 왜곡 가능 |
| **Category 타입** | XGBoost가 내부적으로 최적 분할 | **가장 효율적** | XGBoost 전용 |

> `enable_categorical=True`로 설정하면, XGBoost가 범주형 변수의 최적 분할을 자동으로 찾습니다.
> OneHot처럼 차원이 늘어나지 않으면서도 순서 왜곡 없이 최적의 분할을 학습합니다.

#### 최종 피처 수

```
원본 피처 (11) + 파생 피처 (34) + Ridge 메타피처 (1) = 46개
```

---

### Cell 7 — XGBoost 학습 (Stage 2)

```python
xgb_params = {
    'n_estimators': 20000,
    'learning_rate': 0.004,
    'max_depth': 9,
    'subsample': 0.78,
    'reg_lambda': 6,
    'reg_alpha': 0.15,
    'colsample_bytree': 0.55,
    'colsample_bynode': 0.65,
    'min_child_weight': 6,
    'tree_method': 'hist',
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse',
    'enable_categorical': True,
}
```

#### 학습 루프 (10-Fold)

```python
for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y)):
    # 원본 데이터 증강 (있을 경우)
    X_train_combined = pd.concat([X_train_fold, X_original_xgb])
    y_train_combined = pd.concat([y_train_fold, y_orig])

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined,
              eval_set=[(X_val, y_val)],  # early stopping용
              verbose=1000)

    oof_predictions_xgb[val_index] = model.predict(X_val)
    test_predictions_xgb.append(model.predict(X_test_xgb))
```

> **Early Stopping**: 20,000 트리까지 학습 가능하지만, 검증 RMSE가 100 라운드 동안 개선되지 않으면 자동 중단
> → 과적합 방지 + 학습 시간 절약

---

### Cell 8 — 최종 결과 & 제출

```python
# test 예측: 10개 fold 모델의 평균
test_xgb_avg = np.mean(test_predictions_xgb, axis=0)

# 제출 파일 생성
submission = sample_sub.copy()
submission[TARGET] = test_xgb_avg
submission.to_csv('submission.csv', index=False)
```

> **왜 fold별 평균을 하는가?**
> 각 fold 모델은 약간 다른 데이터로 학습되어 "약간 다른 시각"을 가집니다.
> 평균을 내면 개별 모델의 과적합이 상쇄되어 더 안정적인 예측이 됩니다.

---

### Cell 9 — 피처 중요도 분석

```python
importance_dict_xgb = model.get_booster().get_score(importance_type='gain')
```

> **gain**: 해당 피처가 분할에 사용되었을 때의 평균 성능 개선량
> gain이 높을수록 모델에 더 중요한 피처

**확인 사항**:
- [ ] `feature_lr_pred` (Ridge 메타피처)가 상위에 있는지 → 메타피처 전략이 효과적이었다는 증거
- [ ] 파생 피처 중 중요도가 0에 가까운 것은 제거 후보

---

## 4. 핵심 개념 딥다이브

### 4.1 OOF (Out-of-Fold) 예측

#### 개념

train의 각 행에 대해 **"그 행을 학습에 사용하지 않은 모델"** 이 예측한 값

#### 왜 필요한가?

| 목적 | 설명 |
|------|------|
| **공정한 성능 비교** | 모든 행이 "검증 상황"에서의 예측 → 과적합 없는 진짜 성능 |
| **메타피처 생성** | 다음 스테이지 모델의 입력으로 사용 (정보 누수 없음) |
| **블렌딩 가중치** | 모델 간 최적 가중치를 OOF 기반으로 계산 |
| **에러 분석** | 어떤 구간/그룹에서 모델이 약한지 행 단위로 분석 가능 |

### 4.2 Stacking vs Blending

이 노트북은 **Stacking (메타피처)** 방식을 사용합니다.

| 구분 | Blending (이전 버전) | Stacking (현재 버전) |
|------|------|------|
| 방법 | `w × Ridge + (1-w) × XGBoost` | Ridge 예측을 XGBoost의 피처로 사용 |
| 유연성 | 선형 조합만 가능 | XGBoost가 비선형으로 활용 가능 |
| 성능 | 제한적 | 더 높은 성능 가능 |
| 복잡도 | 단순 | 약간 복잡 |

### 4.3 TargetEncoder vs OneHotEncoder vs OrdinalEncoder

| 방식 | 원리 | 적합한 모델 | 주의점 |
|------|------|------------|--------|
| **OneHot** | 카테고리마다 0/1 열 생성 | 선형 모델 | 고카디널리티 시 차원 폭발 |
| **Ordinal** | 카테고리를 정수로 변환 | 트리 모델 | 인위적 순서 부여 |
| **Target** | 카테고리를 타겟 평균으로 변환 | 모든 모델 | 반드시 CV 내부에서 fit해야 누수 방지 |

> 이 노트북에서 Ridge에 TargetEncoder를 사용한 이유:
> - Ridge는 선형 모델이므로 숫자 입력이 필요
> - TargetEncoder는 타겟과의 관계를 직접 반영하므로 선형 모델과 궁합이 좋음
> - `smooth='auto'`로 적은 샘플의 카테고리에서 과적합을 방지

### 4.4 Cross-Validation (교차 검증)

```python
kf = KFold(n_splits=10, shuffle=True, random_state=1003)
```

| 설정 | 값 | 이유 |
|------|-----|------|
| `n_splits` | 10 | 안정적인 평가 (5보다 분산이 낮음) |
| `shuffle` | True | 데이터 순서에 의한 편향 방지 |
| `random_state` | 1003 | 재현성 확보 (Ridge와 XGBoost가 같은 분할 사용) |

> **10-Fold를 선택한 이유**: 데이터가 63만 행으로 충분히 크므로,
> 각 fold에 6.3만 행이 검증 셋이 되어도 통계적으로 안정적입니다.

---

## 5. 피처 엔지니어링 총정리

### 5.1 변환 유형별 목적과 효과

#### 다항식 변환 (Polynomial)

```python
df['study_hours_squared'] = df['study_hours'] ** 2
```

- **목적**: "공부를 조금 더 하면 점수가 크게 오르는" 비선형 효과 포착
- **직관**: 공부 1→2시간은 효과가 크고, 8→9시간은 효과가 작을 수 있음 (수확 체감)

#### 로그 변환 (Log)

```python
df['log_study_hours'] = np.log1p(df['study_hours'].clip(lower=0))
```

- **목적**: 오른쪽으로 치우친 분포를 정규분포에 가깝게 변환
- **효과**: 이상치의 영향 완화, 선형 모델 성능 개선

#### 상호작용 (Interaction)

```python
df['study_hours_times_attendance'] = df['study_hours'] * df['class_attendance']
```

- **목적**: "두 변수가 동시에 높을 때" 시너지 효과 포착
- **직관**: 공부를 많이 하면서(study_hours↑) 수업도 잘 출석하면(attendance↑) 시너지 효과

#### 비율 (Ratio)

```python
df['study_hours_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
```

- **목적**: 절대값이 아닌 상대적 비교
- **직관**: 같은 공부 시간이라도 수면 시간 대비 비율이 다르면 효과가 다를 수 있음

#### 규칙 기반 플래그 (Rule-based Flag)

```python
df['ideal_sleep_flag'] = ((df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9)).astype(int)
```

- **목적**: 도메인 지식을 피처로 변환
- **직관**: 수면 과학에서 7~9시간이 최적이라는 지식을 모델에 직접 알려줌

#### 갭 피처 (Gap)

```python
df['sleep_gap_8'] = (df['sleep_hours'] - 8.0).abs()
```

- **목적**: "이상적 값에서 얼마나 벗어났는가"를 수치화
- **직관**: 8시간 수면이 이상적이라면, 5시간이든 11시간이든 8에서 멀수록 불리

### 5.2 최종 피처 구성

```
┌──────────────────────────────────────────┐
│           총 46개 피처                     │
├──────────────────────────────────────────┤
│  원본 피처          │  11개               │
│  ├─ 수치형         │   4개               │
│  └─ 범주형         │   7개               │
├──────────────────────────────────────────┤
│  파생 피처          │  34개               │
│  ├─ 다항식(제곱)    │   4개               │
│  ├─ 로그 변환      │   3개               │
│  ├─ 제곱근 변환    │   2개               │
│  ├─ 수치 상호작용  │   4개               │
│  ├─ 비율           │   3개               │
│  ├─ 순서형 인코딩  │   3개               │
│  ├─ 순서×수치      │   3개               │
│  ├─ 순서×순서      │   2개               │
│  ├─ 플래그         │   3개               │
│  ├─ 효율 지표      │   1개               │
│  ├─ 갭 피처        │   2개               │
│  └─ 구간화         │   4개               │
├──────────────────────────────────────────┤
│  메타피처 (Ridge)   │   1개               │
├──────────────────────────────────────────┤
│  XGBoost 입력 합계  │  46개 (+ 메타 1)     │
└──────────────────────────────────────────┘
```

---

## 6. 모델링 전략 상세

### 6.1 Stage 1: RidgeCV (메타피처 생성기)

#### Ridge 회귀란?

```
일반 선형 회귀:  min Σ(y - Xw)²
Ridge 회귀:     min Σ(y - Xw)² + α‖w‖²
                                   ↑ L2 정규화 항
```

- `α` (alpha)가 클수록: 계수(w)가 작아짐 → 단순한 모델 → 과적합 방지
- `α`가 작을수록: 일반 선형 회귀에 가까움 → 복잡한 모델 → 과적합 위험

#### RidgeCV의 alpha 탐색

```python
alphas = np.logspace(-3, 3, 20)
# [0.001, 0.002, ..., 1.0, ..., 100, 1000]
```

> 20개 후보 중 5-fold CV로 최적 alpha를 자동 선택

#### Ridge가 메타피처로 적합한 이유

1. **빠르다**: 학습 시간이 짧아 10-fold × 5-fold CV도 부담 없음
2. **안정적**: 과적합이 적어 OOF 예측이 안정적
3. **보완적**: XGBoost와 다른 관점(선형 vs 비선형)의 정보를 제공

### 6.2 Stage 2: XGBoost (최종 예측기)

#### XGBoost란?

**eXtreme Gradient Boosting** — 순차적으로 약한 트리를 추가하면서 이전 트리의 잔차를 학습

```
예측 = Tree₁(x) + Tree₂(x) + Tree₃(x) + ... + Treeₙ(x)
       ↑ 전체 경향   ↑ 잔차 보정   ↑ 더 세밀한 보정
```

#### 원본 데이터 증강 (Data Augmentation)

```python
if X_original_xgb is not None:
    X_train_combined = pd.concat([X_train_fold, X_original_xgb])
    y_train_combined = pd.concat([y_train_fold, y_orig])
```

> Kaggle Playground Series의 합성 데이터 + 원본 데이터를 합쳐서 학습
> → 모델이 더 다양한 패턴을 학습하여 일반화 성능 향상

#### Early Stopping 동작 원리

```
Round 1000:  val RMSE = 8.85
Round 1100:  val RMSE = 8.83  ← 개선됨, 카운터 리셋
Round 1200:  val RMSE = 8.84  ← 악화 시작, 카운터 1
...
Round 1300:  val RMSE = 8.86  ← 100라운드 동안 미개선 → 중단!
→ 최적 모델 = Round 1100의 모델 사용
```

---

## 7. 하이퍼파라미터 해부

### XGBoost 파라미터 상세

| 파라미터 | 값 | 의미 | 조정 가이드 |
|---------|-----|------|------------|
| `n_estimators` | 20,000 | 최대 트리 수 | early stopping과 함께 사용하므로 크게 설정 |
| `learning_rate` | 0.004 | 각 트리의 기여도 | 작을수록 정밀하지만 더 많은 트리 필요 |
| `max_depth` | 9 | 트리 깊이 | 클수록 복잡한 패턴, 과적합 위험↑ |
| `subsample` | 0.78 | 행 샘플링 비율 | 각 트리에 78%의 데이터만 사용 → 과적합 방지 |
| `colsample_bytree` | 0.55 | 트리별 열 샘플링 | 각 트리에 55%의 피처만 사용 |
| `colsample_bynode` | 0.65 | 노드별 열 샘플링 | 각 분할에 65%의 피처만 고려 |
| `reg_lambda` | 6 | L2 정규화 | 큰 값 → 계수 축소 → 과적합 방지 |
| `reg_alpha` | 0.15 | L1 정규화 | 작은 값 → 약한 희소성 유도 |
| `min_child_weight` | 6 | 최소 리프 가중치 | 클수록 보수적 분할 → 과적합 방지 |
| `tree_method` | 'hist' | 히스토그램 기반 | 대용량 데이터에서 빠름 |
| `early_stopping_rounds` | 100 | 조기 중단 기준 | 100라운드 미개선 시 중단 |
| `enable_categorical` | True | 범주형 지원 | category 타입 직접 처리 |

### 파라미터 간 상호작용

```
learning_rate ↓ + n_estimators ↑ = 정밀하지만 느린 학습
  → early_stopping으로 적정선에서 자동 중단

subsample + colsample = 랜덤성 도입
  → 각 트리가 데이터의 다른 부분을 학습
  → 앙상블 효과로 과적합 감소

max_depth + min_child_weight = 트리 복잡도 제어
  → depth가 커도 min_child_weight가 크면 보수적
```

---

## 8. 버전별 발전 과정

### 스코어 히스토리

| 버전 | Kaggle Score | 전략 | 핵심 변화 |
|------|-------------|------|----------|
| v1 | 8.78741 | Ridge + HistGB 블렌딩 | 기본 파이프라인 구축 |
| v2 | 8.75609 | 4모델 블렌딩 + Optuna | XGBoost/LightGBM 추가, 하이퍼파라미터 최적화 |
| v3 | — | v2 개선 | 코드 정리, Optuna 75 trials, 피처 중요도 분석 |
| **v4 (현재)** | — | **Ridge 메타피처 + XGBoost** | **46개 피처 엔지니어링, 스태킹** |

### 전략 진화 요약

```
v1: 단순 블렌딩
    Ridge ──┐
            ├──→ 가중 평균 → 제출
    HistGB ─┘

v2-v3: 멀티모델 블렌딩
    Ridge    ──┐
    HistGB   ──┤
    XGBoost  ──┼──→ scipy.optimize → 최적 가중치 → 제출
    LightGBM ──┘

v4 (현재): 스태킹 (메타피처)
    Ridge ──→ 예측값 ──→ ┐
                          ├──→ XGBoost ──→ 제출
    46개 피처 ────────→   ┘
```

### v1 (초기 버전) 주요 학습 포인트

- **Pipeline + ColumnTransformer**: CV 누수를 원천 차단하는 전처리 패턴
- **OOF 기반 블렌딩**: 2개 모델의 가중 평균 (Ridge 12% + HistGB 88%)
- **블렌딩 가중치 수식**:

```
diff = pred_a - pred_b
w = (y - pred_b) · diff / (diff · diff)
w = clip(w, 0, 1)
```

> 분모: 두 모델 예측의 차이 에너지
> 분자: pred_b에서 y로 가려면 diff 방향으로 얼마나 이동해야 하는지

### v2 주요 학습 포인트

- **모델 다양성**: Ridge, HistGB, XGBoost, LightGBM 4개 모델
- **Optuna 하이퍼파라미터 최적화**: 10 trials
- **파생 피처 추가**: study_efficiency, sleep_study_ratio 등 4개
- **scipy.optimize 블렌딩**: N개 모델 가중치를 최적화 (합=1 제약)

### v3 주요 학습 포인트

- **코드 간소화**: 불필요한 try-except, 조건부 로직 제거
- **Optuna 75 trials**: 더 나은 하이퍼파라미터 탐색
- **RandomForest 기반 피처 중요도 분석**: 중요도 < 0.01인 피처 자동 제거

### v4 (현재) 주요 발전

| 항목 | 이전 버전 | 현재 버전 |
|------|----------|----------|
| 피처 수 | 11 + 4 = 15 | 11 + 34 = 45 |
| 범주형 인코딩 | OHE/Ordinal | TargetEncoder + Category native |
| 모델 결합 | 가중 평균 (블렌딩) | 메타피처 (스태킹) |
| 최종 모델 | 4모델 블렌딩 | XGBoost 단일 (Ridge 피처 포함) |
| CV | 5-Fold | 10-Fold |
| 데이터 증강 | 없음 | 원본 데이터 결합 |

---

## 9. 자기 점검 질문 15선

아래 질문에 답할 수 있으면 이 노트북을 **완전히 내 것으로 만든 것**입니다.

### 기본 개념 (1~5)

1. **OOF 예측이란 무엇이고, 왜 스태킹/블렌딩에 필수인가?**
   - 힌트: "누수 없는 train 예측"

2. **`assert TARGET not in test_df.columns`는 어떤 사고를 막는가?**
   - 힌트: 데이터 누수

3. **Ridge에 TargetEncoder를 사용한 이유는? OneHot 대비 장점은?**
   - 힌트: 타겟 관계 직접 반영, 차원 축소

4. **`np.log1p(x)`가 `np.log(x)`보다 안전한 이유는?**
   - 힌트: x=0일 때

5. **test 예측을 fold별로 평균내는 이유는?**
   - 힌트: 과적합 상쇄, 앙상블 효과

### 심화 개념 (6~10)

6. **블렌딩(가중 평균)과 스태킹(메타피처)의 차이는? 어떤 것이 왜 더 유연한가?**
   - 힌트: 선형 조합 vs 비선형 활용

7. **TargetEncoder를 fold 밖에서 fit하면 어떤 문제가 생기나?**
   - 힌트: 타겟 정보 누수 → 과적합

8. **`early_stopping_rounds=100`과 `n_estimators=20000`의 관계는?**
   - 힌트: 넉넉한 예산 + 자동 중단

9. **`subsample`과 `colsample_bytree`가 동시에 1보다 작으면 어떤 효과가 있나?**
   - 힌트: 행/열 모두 랜덤 샘플링 → 배깅 효과

10. **Ridge 메타피처의 중요도가 높다면, 이것은 무엇을 의미하는가?**
    - 힌트: "선형 모델이 보는 세계"가 XGBoost에게 유용한 추가 정보

### 실전 응용 (11~15)

11. **도메인 지식 기반 플래그(`ideal_sleep_flag`)는 왜 만드나? 트리 모델이 스스로 찾을 수 없나?**
    - 힌트: 트리도 찾을 수 있지만, 미리 만들면 학습이 더 쉬워짐

12. **`clip(0, 100)`은 왜 필요한가? 없으면 어떤 일이 생기나?**
    - 힌트: 물리적 범위 외의 예측 → 메타피처가 오염될 수 있음

13. **원본 데이터(Original Data)를 증강에 사용하는 것의 장단점은?**
    - 힌트: 장점=다양성, 단점=분포 차이 시 성능 저하

14. **KFold 대신 StratifiedKFold를 쓸 필요가 있나? (회귀 문제에서)**
    - 힌트: 회귀는 연속형이라 일반 KFold가 충분, 단 타겟 분포가 극단적이면 고려

15. **이 파이프라인에서 추가로 성능을 올리려면 무엇을 시도할 수 있나?**
    - 힌트: LightGBM/CatBoost 추가 스태킹, pseudo labeling, 피처 선택 등

---

## 10. 추가 학습 방향

### 즉시 시도해볼 수 있는 개선

- [ ] **LightGBM/CatBoost 메타피처 추가**: Ridge 외에 다른 모델의 OOF도 메타피처로 추가
- [ ] **2nd Level Stacking**: XGBoost 외에 2차 메타러너(예: 선형 모델) 추가
- [ ] **Target Encoding 변형**: CatBoost의 Ordered Target Encoding 시도
- [ ] **피처 선택**: 중요도 낮은 피처 제거 후 성능 비교

### 더 깊이 공부할 주제

- [ ] **Stacking 이론**: Wolpert (1992)의 Stacked Generalization 논문
- [ ] **Feature Engineering 자동화**: Featuretools, AutoFeat 라이브러리
- [ ] **XGBoost 내부 구조**: 2차 Taylor 전개 기반 목적 함수
- [ ] **정규화 이론**: L1(Lasso) vs L2(Ridge) vs ElasticNet의 수학적 차이

### 실전 캐글 스킬

- [ ] **Pseudo Labeling**: test 예측값 중 확신도 높은 것을 train에 추가
- [ ] **Post-Processing**: 예측값의 분포를 train 타겟 분포에 맞추기
- [ ] **Seed Averaging**: 서로 다른 random_state로 학습한 모델들의 평균

---

## 부록: 주요 코드 패턴 레퍼런스

### A. 안전한 경로 탐색 패턴

```python
def find_data_paths():
    """Kaggle/로컬 환경 자동 감지"""
    kaggle_path = Path('/kaggle/input/...')
    if kaggle_path.exists():
        return kaggle_path
    # 로컬 후보 탐색
    for parent in Path.cwd().parents:
        candidate = parent / 'data'
        if candidate.exists():
            return candidate
```

### B. OOF 생성 패턴

```python
oof = np.zeros(len(X))
test_preds = np.zeros((len(X_test), n_folds))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    model.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = model.predict(X[va_idx])      # OOF
    test_preds[:, fold] = model.predict(X_test)  # test fold별

final_test = test_preds.mean(axis=1)  # fold 평균
```

### C. TargetEncoder + CV 안전 패턴

```python
for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    te = TargetEncoder(smooth='auto')
    X_tr[CATS] = te.fit_transform(X_tr[CATS], y_tr)  # train만 fit
    X_va[CATS] = te.transform(X_va[CATS])             # valid는 transform만
    X_test_enc[CATS] = te.transform(X_test[CATS])     # test도 transform만
```

### D. XGBoost + Early Stopping 패턴

```python
model = xgb.XGBRegressor(
    n_estimators=20000,        # 넉넉하게
    early_stopping_rounds=100, # 자동 중단
    eval_metric='rmse'
)
model.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          verbose=1000)
```

---

> **마지막 팁**: 이 노트북의 가장 큰 교훈은, 모델 자체보다 **피처 엔지니어링**과 **메타피처 전략**이 성능을 결정한다는 것입니다. 46개의 잘 설계된 피처 + Ridge 메타피처가 XGBoost의 성능을 극대화하는 핵심이었습니다.
