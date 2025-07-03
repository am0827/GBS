import streamlit as st
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer, util
import torch

# ---- 구글 시트 연동 ---- #
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], scope
)
client = gspread.authorize(creds)
sheet = client.open("ap jeongbo alzar takkarsenn").sheet1

# ---- Streamlit 웹앱 UI ---- #
st.set_page_config(page_title="문학 작품 추천 입력", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        color: white;
        background-color: #6c63ff;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 사용자 참여형 문학 작품 추천 써-비스 : 알자르 타카르센(alzar takkarrsen)")
st.markdown("""
이 플랫폼은 다양한 사용자가 지적 문학 작품을 추천하고, 그 추천 이유와 감정을 함께 기록함으로써 집단 지성 기반의 문학 데이터베이스를 구축한 뒤, AI를 통해 문학 작품을 추천받을 수 있도록 하는 플랫폼입니다.
""")

st.header("✍️ 독서 기록 입력 흐름")
with st.form("book_form"):
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("작품명*")
        author = st.text_input("저자*")
        country = st.text_input("국가")
        period = st.text_input("시대")
    with col2:
        genre = st.text_input("장르* (예: 소설, 시, 희곡, 산문 등)")
        emotion = st.text_input("감정* (예: 고독, 희망, 슬픔 등 — 쉼표로 여러 감정 입력 가능)")
        user = st.text_input("닉네임 (선택)")

    opinion = st.text_area("평가*", height=150)
    submit = st.form_submit_button("📤 독서 기록 제출")

    if submit:
        if title and author and opinion:
            row = [title, author, country, period, genre, emotion, opinion, user]
            try:
                sheet.append_row(row)
                st.success("✅ 독서 기록이 저장되었습니다.")
            except Exception as e:
                st.error(f"❌ 저장 중 오류 발생: {e}")
        else:
            st.warning("⚠️ 작품명, 저자, 평가는 필수 입력 항목입니다.")

# ---- 경과 보기 ---- #
st.header("📄 경과 보기")
try:
    data = sheet.get_all_records()
    df_recent = pd.DataFrame(data)
    if not df_recent.empty:
        st.dataframe(df_recent[::-1], use_container_width=True)
    else:
        st.info("아직 입력된 작품이 없습니다.")
except:
    st.error("Google Sheets 데이터를 불러오려면 API 키 또는 시트 공유 설정을 확인해주세요.")

# ---- AI 참조 받기 ---- #
st.header("🔎 문학 작품 AI 참조 받기")

@st.cache_data(ttl=600)
def load_data():
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    df.fillna("", inplace=True)
    # 쉼표는 공백으로 변경 (키워드 매칭 용이)
    df["감정"] = df["감정"].astype(str).str.replace(",", " ")
    df["combined_text"] = ("장르: " + df["장르"] + " 감정: " + df["감정"] + " 평가: " + df["평가"])
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

df = load_data()
model = load_model()

query = st.text_input("추천받고 싶은 키워드나 감정을 입력하세요 (쉼표로 여러 개 입력 가능)")

if query:
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        query_list = [q.strip() for q in query.split(",")]

        # query 문장 임베딩 (tensor)
        query_embs = model.encode(query_list, convert_to_tensor=True)
        # 평균 벡터 (tensor)
        avg_query_emb = torch.mean(query_embs, dim=0, keepdim=True)

        # 문서 임베딩 (tensor)
        doc_embs = model.encode(df["combined_text"].tolist(), convert_to_tensor=True)

        # 코사인 유사도 (tensor)
        cos_scores = util.pytorch_cos_sim(avg_query_emb, doc_embs)[0]

        # numpy 배열로 변환
        sims = cos_scores.cpu().numpy()

        df["유사도"] = sims

        # 키워드 점수 계산 (감정, 장르, 평가 모두 문자열 포함 여부 확인)
        df["키워드점수"] = 0
        for kw in query_list:
            df["키워드점수"] += df["감정"].str.contains(kw, case=False, na=False) * 1.0
            df["키워드점수"] += df["장르"].str.contains(kw, case=False, na=False) * 1.0
            df["키워드점수"] += df["평가"].str.contains(kw, case=False, na=False) * 1.0

        # 최종 점수 계산: 유사도 반영 비율 0.7, 키워드 반영 비율 0.3
        df["최종점수"] = df["유사도"] * 0.7 + df["키워드점수"] * 0.3

        df_sorted = df.sort_values(by="최종점수", ascending=False)

        top_n = min(5, len(df_sorted))
        st.write(f"🔍 알자르 타카르센의 추천 작품 {top_n}건:")

        for _, row in df_sorted.head(top_n).iterrows():
            st.markdown(f"### {row['작품명']} - {row['저자']}")
            st.write(f"- **장르**: {row['장르']}  |  **감정**: {row['감정']}")
            st.write(f"- **평가**: {row['평가']}")
            st.write(f"- **유사도**: {row['유사도']:.3f} | **키워드점수**: {row['키워드점수']:.2f} | **최종점수**: {row['최종점수']:.3f}")
            st.markdown("---")
    else:
        st.warning("⚠️ 데이터가 비어있어 추천을 실행할 수 없습니다. 작품을 한 개 이상 먼저 입력해주세요.")
