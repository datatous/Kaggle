### 전체 한 장 요약(뼈대)
이 노트북은 **(1) 경로/데이터 로드 안정화 → (2) 데이터 품질 점검(누수 방지 포함) → (3) EDA로 컬럼 타입/분포/상관 파악 → (4) 모델별 전처리 파이프라인 구성 → (5) KFold로 OOF(out-of-fold) 예측 생성 → (6) OOF 기반으로 블렌딩 가중치 최적화 → (7) test 예측으로 제출파일 생성** 흐름입니다.

---

## 1) “셀 순서대로” 코드-학습용 가이드 (입력/출력/핵심 포인트)

아래는 **각 셀을 실행하면서 무엇을 확인해야 하는지**를 “체크리스트”로 만든 것입니다. (노트북을 옆에 켜고 그대로 대조하며 보면 학습이 가장 빨라요)

### Cell 0 — 문제 정의/전략 선언(설명 셀)
- **무엇**: 대회명, 타겟(`exam_score`), 지표(RMSE), OOF 기반 선택 방침
- **왜**: “무슨 모델이든” 결국 **평가 기준과 선택 기준**이 흔들리면 성능이 안 나옴
- **체크**:
  - 타겟 컬럼명이 맞는지: `exam_score`
  - 로컬 검증을 RMSE로 할지(대회 metric과 동일해야 비교 가능)

---

### Cell 1 — 환경 세팅(재현성 + 한글 폰트 + 데이터 경로 자동탐색)
#### 1) 재현성
- `RANDOM_STATE = 42`, `np.random.seed(42)`
- **왜**: CV split, 모델의 랜덤 요소가 고정되어 실험 비교가 가능

#### 2) Matplotlib 한글 폰트
- Windows면 `Malgun Gothic`
- `plt.rc('axes', unicode_minus=False)`로 마이너스 기호 깨짐 방지
- **체크**: 그래프 제목/축 라벨이 한글로 정상 출력되는지

#### 3) `find_data_dir()` (실전에서 매우 유용)
- **무엇**: 여러 후보 경로를 돌며 `train.csv`가 있는 `data` 폴더를 자동 탐색
- **왜**:
  - 노트북 실행 위치(working directory)가 바뀌면 상대경로가 깨지기 쉬움
  - Kaggle/로컬/IDE 환경이 바뀌어도 “그냥 돌아가게” 만듦
- **체크**:
  - 출력된 `TRAIN_PATH.exists()`가 True인지
  - 출력된 `DATA_DIR.resolve()`가 실제 폴더인지

> 공부 포인트: “데이터 분석 노트북”은 모델보다 **입력 경로/실행 안정성** 때문에 실패하는 일이 많습니다. 이 셀은 그 문제를 원천 차단하는 패턴이에요.

---

### Cell 2 — 데이터 로드(try-except로 안전하게)
- `pd.read_csv(TRAIN_PATH)`, `pd.read_csv(TEST_PATH)`, `pd.read_csv(SAMPLE_SUB_PATH)`
- `FileNotFoundError` 시 친절한 에러 메시지로 경로 출력
- **체크**:
  - `train: (630000, 13) test: (270000, 12)` 처럼 크기가 기대와 같은지
  - `head()`로 컬럼들이 올바른지(특히 `id`, `exam_score`)

> 공부 포인트: 대회에서 test에는 타겟이 없어야 정상입니다.

---

### Cell 3 — 데이터 품질 체크(EDA의 0단계)
#### 1) 누수 방지 assert
- `TARGET_COL='exam_score'`가 train에 있는지
- test에 타겟이 없는지
- **왜**: 이거 한 줄이 “실수로 타겟이 포함된 test” 같은 치명적 누수를 잡습니다.

#### 2) dtypes / missing / duplicates
- `dtypes`: 범주형 vs 수치형 분리의 근거
- `isna().mean()`: 결측률 파악 → imputer 전략 결정
- `duplicated().sum()`: 중복이 많으면 분포가 왜곡될 수 있음
- **체크**:
  - 결측률이 큰 컬럼이 있는지(전처리에서 반드시 처리 필요)
  - 중복행이 0인지(출력에서 0)

#### 3) 타겟 분포
- `sns.histplot(exam_score, kde=True)`
- **왜**:
  - 타겟이 치우쳐 있으면 변환/클리핑/로버스트 손실 등을 고려
  - 모델이 “평균 회귀”만 해도 어느 정도 되는지 감 잡기

---

### Cell 4 — 간단 EDA(컬럼 분류/요약/상관)
#### 1) feature/target 분리
- `feature_cols = train_df.columns - {exam_score}`
- `X=train[feature_cols]`, `y=train[exam_score]`, `X_test=test[feature_cols]`

#### 2) 타입 분리 로직
- `categorical_cols = X.select_dtypes(include=['object'])`
- `numeric_cols = 나머지`
- **출력상**:
  - numeric: `['id','age','study_hours','class_attendance','sleep_hours']`
  - categorical: `['gender','course','internet_access','sleep_quality','study_method','facility_rating','exam_difficulty']`

> 공부 포인트(중요): 여기서 **전처리 전략이 결정**됩니다.  
> - 범주형이 많고 카디널리티가 크면 원핫은 폭발할 수 있음  
> - 트리 모델은 ordinal이 훨씬 효율적인 경우가 많음

#### 3) 요약 통계/카디널리티
- 수치형 `describe()`로 스케일/이상치 감
- 범주형 `nunique()`로 고유값 개수 확인
- **체크**:
  - `id`가 feature에 포함되어 있는데, 이게 의미 있는 신호인지(대회에 따라 “의미 없는 키”일 수도, “시간/순서 신호”일 수도 있음)

#### 4) 상관 히트맵(샘플링)
- 63만 행 전부로 corr을 구하기보다 5만 샘플로 빠르게 확인
- **왜**: 탐색 단계에서는 “정확성”보다 “방향성/가설”이 중요

---

## 2) 전처리/OOF/블렌딩 “깊게” (이 노트북의 성능 핵심)

### Cell 5 — 전처리 파이프라인(모델별로 다르게 설계한 이유)
이 셀은 “좋은 캐글 노트북”의 전형입니다. **전처리를 데이터프레임에서 미리 해두지 않고**, 모델과 함께 `Pipeline`에 넣어 **CV 누수를 원천 차단**합니다.

#### (A) 왜 Pipeline + ColumnTransformer인가?
- **문제**: 전처리를 전체 train에 한 번에 fit하면, CV의 valid 폴드 정보가 전처리에 섞여 들어갈 수 있음(미묘하지만 실제로 성능 뻥튀기 가능)
- **해결**: `model.fit(X_tr, y_tr)`를 할 때 전처리도 X_tr에만 fit → valid는 transform만
- **결론**: CV 점수가 더 “정직”해짐

#### (B) 수치형 전처리: `median imputer`
- `SimpleImputer(strategy='median')`
- **왜 median?**
  - 평균보다 이상치에 덜 민감
  - 트리/선형 모두 무난

#### (C) 범주형 전처리 2가지: OHE vs Ordinal
1) **Ridge(선형)** → `OneHotEncoder`
- 선형 모델은 범주를 숫자로 그냥 넣으면 “순서”로 오해할 수 있음  
- 원핫은 각 카테고리를 독립 차원으로 만들어 선형이 잘 학습
- `handle_unknown='ignore'`: test에 새로운 카테고리 나와도 에러 안 남

2) **HistGradientBoosting(트리)** → `OrdinalEncoder(unknown=-1)`
- 트리 계열은 원핫도 가능하지만, 대규모 데이터에서 원핫은 차원이 너무 커질 수 있음
- Ordinal은 훨씬 가볍고 빠름
- `unknown_value=-1`은 “미지 카테고리”를 한 값으로 모아 안정화

> 주의(공부 포인트): Ordinal은 “순서 의미”를 만들지만, 트리는 분기 규칙을 학습하면서 이를 활용할 수 있어서 실무/캐글에서 자주 쓰입니다. 다만 데이터/모델에 따라 원핫이 더 나을 수도 있어 비교가 중요합니다(이 노트북은 둘 다 비교).

#### (D) StandardScaler(with_mean=False)의 이유(매우 중요)
- 원핫 결과는 **희소행렬(sparse)** 인 경우가 많음
- 평균을 빼면 0이 아닌 값이 대거 생겨 **dense**가 되어 메모리가 폭발할 수 있음
- 그래서 `with_mean=False`

#### (E) 모델 선택 이유(직관)
- `Ridge`: 선형 베이스라인으로 강력 + 빠름 + 안정적
- `HistGradientBoostingRegressor`: 스킷런 내장 GBDT 계열로 빠르고 성능 좋음(특히 대용량에 유리)

---

### Cell 6 — CV + OOF 생성(왜 이렇게 짜는가)
#### (A) OOF(out-of-fold)란?
- train의 각 행에 대해 **“그 행을 학습에 사용하지 않은 모델”** 로 예측한 값
- 즉, train 전체에 대해 “검증 상황의 예측치”를 만든 것

#### (B) 왜 OOF가 성능 비교에 가장 공정한가?
- 단순히 fold 평균 RMSE만 보면, “각 fold에서 무엇이 예측됐는지”가 남지 않음
- OOF는 행 단위 예측이 남기 때문에:
  - 모델 A/B가 **어떤 구간에서** 강한지 비교 가능
  - 블렌딩/스태킹에 바로 사용 가능

#### (C) 코드 흐름 해부
- `cv = KFold(shuffle=True, random_state=42)`
- `oof_pred = np.zeros(len(X))` (train 예측 저장)
- `test_pred = np.zeros(len(X_test))` (test 예측을 fold별로 평균)
- 루프에서:
  - `model.fit(X_tr, y_tr)`
  - `va_pred = model.predict(X_va)`
  - `oof_pred[va_idx] = va_pred`  ← OOF 완성의 핵심
  - `test_pred += model.predict(X_test)/n_splits`  ← fold 평균

#### (D) “OOF RMSE” 출력이 중요한 이유
- fold 평균 RMSE와 OOF RMSE는 보통 같거나 매우 유사
- OOF RMSE는 “전체 train에 대한 한 번의 예측”이므로, 블렌딩 가중치 계산 같은 후처리에 쓰기 좋음

---

### Cell 7 — 두 모델의 OOF 성능 비교(해석 포인트)
출력에서:
- Ridge OOF RMSE ≈ 8.8948
- HistGB OOF RMSE ≈ 8.8201

**해석**
- 트리 모델이 더 잘 맞는 비선형/상호작용(예: 공부시간×수면×출석 같은 조합)을 잡았을 가능성이 큼
- Ridge는 “전체 평균적 경향”을 안정적으로 잡아주지만, 복잡한 패턴은 약할 수 있음

---

### Cell 8 — OOF 기반 최적 블렌딩(수식/직관/주의점)
#### (A) 목표
- 최종 예측: `blend = w * ridge + (1-w) * hgb`
- w를 “감”이 아니라 **OOF에서 MSE 최소화**로 계산

#### (B) 왜 MSE 최소화가 RMSE에도 유효한가?
- RMSE는 \(\sqrt{MSE}\)라서, **MSE가 작아지면 RMSE도 작아짐**
- 따라서 MSE 최소화로 w를 구해도 RMSE 개선으로 이어지는 경우가 많음

#### (C) `optimal_blend_weight` 수식 의미(직관)
- `diff = pred_a - pred_b`
- 분모 `diff·diff`는 두 모델 예측이 얼마나 다른지(차이의 에너지)
- 분자 `(y - pred_b)·diff`는 “pred_b에서 y로 가려면 diff 방향으로 얼마나 가야 하는지”
- 그 결과 w가 0이면 전부 b, 1이면 전부 a가 최적이라는 뜻

#### (D) 왜 clip(0,1)을 하는가?
- 이론상 최적해가 0~1 밖으로 나올 수도 있음(특히 노이즈/과적합/상관 구조 때문에)
- 하지만 실전에서는 음수 가중치/과도한 외삽이 오히려 불안정해져서
- **안정성 위해 0~1로 제한** (캐글에서 매우 흔한 안전장치)

#### (E) 결과 해석
- `w_ridge ≈ 0.1176`, `w_hgb ≈ 0.8824`
- 즉, HistGB가 주력이고 Ridge는 보정 역할
- Blend OOF RMSE가 HistGB보다 소폭 개선 → 두 모델이 “조금 다른 실수”를 하고 있었다는 의미(블렌딩 이득)

#### (F) KDE sanity check를 왜 그리나?
- 블렌딩이 특정 구간에서 과하게 치우치거나(너무 좁거나/너무 넓거나)
- 이상치 예측이 폭증하면 위험 신호
- 분포 비교로 빠르게 “상식적”인지 확인

---

### Cell 9 — 제출 생성(대회 실수 방지 포인트)
#### (A) sample_submission을 복사하는 이유
- 대회마다 예측 컬럼명이 다름(예: `target`, `exam_score`, `score` 등)
- 그래서 `sample_sub.copy()` 후, `id` 제외 컬럼을 찾아 그 칼럼에 예측을 넣음
- `assert len(pred_col) == 1`로 포맷 이상을 조기 탐지

#### (B) 저장 위치
- `out_path = DATA_DIR.parent / 'submission.csv'`
- data 폴더가 아니라 프로젝트 폴더에 저장해서 관리가 편함

#### (C) 제출 전 최종 체크
- `id` 중복이 0인지
- `(270000, 2)` 형태인지

---

## 3) “공부를 완성”하기 위한 핵심 질문 10개(스스로 점검용)
아래 질문에 답할 수 있으면 이 노트북을 **완전히 내 것으로 만든 것**입니다.

1) 왜 `find_data_dir()` 같은 경로 탐색이 실전에서 중요한가?  
2) `assert TARGET_COL not in test_df.columns`는 어떤 사고를 막는가?  
3) 결측률을 본 뒤 왜 수치형은 median, 범주형은 most_frequent를 썼나?  
4) Ridge에는 왜 OneHot이 유리하고, 트리에는 왜 Ordinal이 실전에서 자주 쓰이나?  
5) `StandardScaler(with_mean=False)`가 왜 필수인가(희소행렬 관점)?  
6) OOF 예측이 무엇이고, 왜 블렌딩/스태킹에 필수인가?  
7) `test_pred`를 fold별 평균내는 이유는 무엇인가?  
8) 왜 블렌딩 가중치를 OOF에서 최적화해야 하는가(누수/과적합 관점)?  
9) `clip(0,1)`은 어떤 불안정성을 막는가?  
10) sample_submission 포맷을 그대로 쓰는 것이 왜 안전한가?

---

## 4) (선택) 너가 더 성장하려면, 이 노트북을 이렇게 확장해볼 수 있어요
- **OOF 기반 진단**: “어떤 구간에서” Ridge가 강한지/HistGB가 강한지  
  - 예: 잔차를 `age` 구간별로 평균내서 모델별 약점을 찾기
- **피처 검토**: `id`가 성능에 진짜 기여하는지(대회에 따라 의미 없을 수 있음)  
- **CV 전략**: KFold 말고 Stratify가 필요할 정도로 타겟 분포가 특이한지(회귀라 보통 KFold)  
- **간단 튜닝**: Ridge의 `alpha`, HistGB의 `max_leaf_nodes/min_samples_leaf/learning_rate` 조금만 스윕해도 성능이 더 좋아질 수 있음

---

원하면 내가 **“셀별로 그대로 복사해서 붙일 수 있는 공부용 주석 버전(한글로 왜/주의점 중심)”** 형태로도 정리해줄게요. 특히 헷갈렸던 지점이 `Cell 5(전처리)` / `Cell 6(OOF)` / `Cell 8(블렌딩 수식)` 중 어디인지 말해주면, 그 파트는 예시(작은 수치 장난감 데이터로 w가 어떻게 계산되는지)까지 붙여서 더 확실히 이해되게 만들어줄게요.