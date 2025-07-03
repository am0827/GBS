import streamlit as st
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

st.title("📚 사용자 참여형 문학 작품 추천 써-비쓰 : 알자르 타카르센(alzar takkarrsen)")
st.markdown("""
이 플랫폼은 다양한 사용자가 직접 문학 작품을 추천하고, 그 추천 이유와 감정을 함께 기록함으로써 집단 지성 기반의 문학 데이터베이스를 구축한 뒤, AI를 통해 문학 작품을 추천받을 수 있도록 하는 플랫폼입니다.
""")

st.header("✍️ 독서 기록 입력 틀")
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

# ---- 최근 추천 작품 보기 ---- #
st.header("📄 최근 입력된 작품")
try:
    data = sheet.get_all_records()
    df_recent = pd.DataFrame(data)
    if not df_recent.empty:
        st.dataframe(df_recent[::-1], use_container_width=True)
    else:
        st.info("아직 입력된 작품이 없습니다.")
except:
    st.error("Google Sheets 데이터를 불러올 수 없습니다. API 키 또는 시트 공유 설정을 확인하세요.")

# ---- 문학 작품 AI 추천 ---- #
st.header("🔎 문학 작품 AI 추천 받기")

@st.cache_data(ttl=600)
def load_data():
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    df.fillna("", inplace=True)
    df["감정"] = df["감정"].astype(str).str.replace(",", " ")
    df["combined_text"] = ("장르: " + df["장르"] + " 감정: " + df["감정"] + " 평가: " + df["평가"])

@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

df = load_data()
model = load_model()

query = st.text_input("추천받고 싶은 키워드나 감정을 입력하세요 (쉼표로 여러 개 입력 가능)")

if query:
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        query_list = [q.strip() for q in query.split(",")]
        query_emb = model.encode(query_list)
        avg_query_emb = query_emb.mean(axis=0).reshape(1, -1)
        doc_embs = model.encode(df["combined_text"].tolist())
        sims = cosine_similarity(avg_query_emb, doc_embs)[0]
        df["유사도"] = sims
        top_n = 5
        results = df.sort_values(by="유사도", ascending=False).head(top_n)

        st.write(f"🔍 알자르 타카르센의 추천 작품 {top_n}건:")
        for _, row in results.iterrows():
            st.markdown(f"### {row['작품명']} - {row['저자']}")
            st.write(f"- **장르**: {row['장르']}  |  **감정**: {row['감정']}")
            st.write(f"- **평가**: {row['평가']}")
            st.write(f"- **유사도 점수**: {row['유사도']:.3f}")
            st.markdown("---")
    else:
        st.warning("⚠️ 데이터가 비어있어 추천을 실행할 수 없습니다. 작품을 한 개 이상 먼저 입력해주세요.")

