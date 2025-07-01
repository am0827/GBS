import streamlit as st
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- 구글 시트 연동 ---- #
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("ap-jeongbo-00cfcfa3b3ce.json", scope)
client = gspread.authorize(creds)
sheet = client.open("ap jeongbo alzar takarrsen").sheet1  # 시트 이름에 맞게 수정

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

st.title("📚 사용자 참여형 문학 작품 추천 써-비쓰 : 알자르 타카르센(alzar takarrsen)")
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
        emotion = st.text_input("감정* (예: 고독, 희망, 슬픔 등)")
        user = st.text_input("닉네임 (선택)")

    reason = st.text_area("추천 이유*", height=150)
    submit = st.form_submit_button("📤 독서 기록 제출")

    if submit:
        if title and author and reason:
            row = [title, author, country, period, genre, emotion, reason, user]
            try:
                sheet.append_row(row)
                st.success("✅ 독서 기록이 저장되었습니다.")
            except Exception as e:
                st.error(f"❌ 저장 중 오류 발생: {e}")
        else:
            st.warning("⚠️ 작품명, 저자, 추천 이유는 필수 입력 항목입니다.")

# ---- 최근 추천 작품 보기 ---- #
st.header("📄 최근 추천된 작품")
try:
    data = sheet.get_all_records()
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df[::-1], use_container_width=True)
    else:
        st.info("아직 추천된 작품이 없습니다.")
except:
    st.error("Google Sheets 데이터를 불러올 수 없습니다. API 키 또는 시트 공유 설정을 확인하세요.")



st.header("🔎 문학 작품 AI 추천 받기")

@st.cache_data(ttl=600)
def load_data():
    try:
        data = sheet.get_all_records()
        if not data:
            st.warning("📭 아직 데이터가 입력되지 않았습니다.")
            st.stop()
        df = pd.DataFrame(data)
        df.columns = [str(col).strip() for col in df.columns]
        df.fillna("", inplace=True)
        st.write("📌 현재 DataFrame 컬럼명 목록:", df.columns.tolist())
        df["combined_text"] = df["추천 이유"] + " " + df["감정"] + " " + df["장르"]
        return df
    except Exception as e:
        st.error(f"❌ 데이터를 불러오는 중 오류 발생: {e}")
        return pd.DataFrame()  # 빈 데이터프레임 반환

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

df = load_data()
model = load_model()

query = st.text_input("추천받고 싶은 키워드나 감정을 입력하세요")

if query:
    query_emb = model.encode([query])
    doc_embs = model.encode(df["combined_text"].tolist())
    sims = cosine_similarity(query_emb, doc_embs)[0]
    df["유사도"] = sims
    top_n = 5
    results = df.sort_values(by="유사도", ascending=False).head(top_n)

    st.write(f"🔍 '{query}'와 가장 관련성 높은 추천 작품 {top_n}건:")
    for _, row in results.iterrows():
        st.markdown(f"### {row['작품명']} - {row['저자']}")
        st.write(f"- **장르**: {row['장르']}  |  **감정**: {row['감정']}")
        st.write(f"- **추천 이유**: {row['추천 이유']}")
        st.write(f"- **유사도 점수**: {row['유사도']:.3f}")
        st.markdown("---")

