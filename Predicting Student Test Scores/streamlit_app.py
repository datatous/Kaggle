import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


# Matplotlib 한글 폰트 설정 (요청사항)
system_name = platform.system()
if system_name == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif system_name == "Darwin":  # Mac
    plt.rc("font", family="AppleGothic")
else:  # Linux (Colab, Docker)
    plt.rc("font", family="NanumGothic")
plt.rc("axes", unicode_minus=False)


DATA_DIR = Path("Predicting Student Test Scores/data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


@st.cache_data(show_spinner=False)
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


st.set_page_config(page_title="Student Test Scores - EDA", layout="wide")
st.title("Student Test Scores (S6E1) - 빠른 EDA 대시보드")

try:
    train_df, test_df = load_data()
except FileNotFoundError:
    st.error(
        "데이터 파일을 찾을 수 없습니다.\n"
        f"- {TRAIN_PATH}\n"
        f"- {TEST_PATH}\n"
    )
    st.stop()

target_col = "exam_score"
id_col = "id"

with st.sidebar:
    st.header("옵션")
    dataset_name = st.radio("데이터", ["train", "test"], index=0)
    df = train_df if dataset_name == "train" else test_df

    all_cols = df.columns.tolist()
    default_col = target_col if target_col in all_cols else all_cols[0]
    col = st.selectbox("컬럼 선택", all_cols, index=all_cols.index(default_col))

    sample_n = st.slider(
        "표본 샘플 크기",
        min_value=5_000,
        max_value=200_000,
        value=min(50_000, len(df)),
        step=5_000,
    )

st.subheader(f"요약: {dataset_name}")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Cols", f"{df.shape[1]:,}")
c3.metric("Missing(%)", f"{(df.isna().mean().mean() * 100):.2f}")

st.dataframe(df.head(30), use_container_width=True)

st.subheader("결측률 (상위 15)")
missing_rate = df.isna().mean().sort_values(ascending=False).head(15)
st.dataframe(missing_rate.to_frame("missing_rate"), use_container_width=True)

st.subheader(f"분포: `{col}`")
plot_df = df.sample(n=min(sample_n, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(8, 4))
if pd.api.types.is_numeric_dtype(plot_df[col]):
    sns.histplot(plot_df[col], bins=50, kde=True, ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(f"{dataset_name} - {col} 분포")
else:
    vc = plot_df[col].astype("string").value_counts().head(30)
    sns.barplot(x=vc.values, y=vc.index, ax=ax)
    ax.set_xlabel("count")
    ax.set_ylabel(col)
    ax.set_title(f"{dataset_name} - {col} 상위 카테고리(Top 30)")

st.pyplot(fig, clear_figure=True)

if dataset_name == "train" and target_col in df.columns:
    st.subheader("수치형 상관관계 (샘플)")
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    corr_df = train_df.sample(
        n=min(sample_n, len(train_df)),
        random_state=42,
    )[num_cols]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Correlation (Sample)")
    st.pyplot(fig, clear_figure=True)

