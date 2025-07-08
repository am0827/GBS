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

# ---- Streamlit 웹 UI ---- #
st.set_page_config(page_title="📚 문학 작품 추천", page_icon="📚", layout="wide")
st.markdown("""
<style>
    .stButton>button { background-color: #6c63ff; color: white; }
    .stTextInput>div>input, .stTextArea>div>textarea { background-color: #fff; }
</style>""", unsafe_allow_html=True)
st.title("📚 사용자 참여형 문학 작품 추천 써-비쓰 : 알자르 탁카르센(Alzar Tahkkarsen)")

# 데이터 입력 폼
with st.form("book_form"):
    title = st.text_input("작품명*")
    author = st.text_input("저자*")
    country = st.text_input("국가")
    period = st.text_input("시대")
    genre = st.text_input("장르*")
    emotion = st.text_input("감정* (쉼표로 여러 감정 입력)")
    opinion = st.text_area("평가*", height=120)
    user = st.text_input("닉네임 (선택)")
    submit = st.form_submit_button("📤 제출")
    if submit:
        if title and author and opinion:
            sheet.append_row([title, author, country, period, genre, emotion, opinion, user])
            st.success("✅ 저장되었습니다!")
        else:
            st.warning("⚠ 작품명, 저자, 평가를 꼭 입력해주세요.")

# 최근 기록 보기
st.header("📄 최근 입력된 작품")
try:
    data = sheet.get_all_records()
    df_recent = pd.DataFrame(data)
    
    if not df_recent.empty:
        st.dataframe(df_recent[::-1], use_container_width=True)
    else:
        st.info("아직 입력된 작품이 없습니다.")
except:
    st.error("시트 데이터 불러오기 실패!")

# AI 추천
st.header("🔎 알잘딱깔센 도서 추천 받기")
@st.cache_data(ttl=600)
def load_data():
    df = pd.DataFrame(sheet.get_all_records())
    if df.empty: return df
    df.columns = [c.strip() for c in df.columns]
    df.fillna("", inplace=True)
    df["감정"] = df["감정"].astype(str).str.replace(",", " ")
    df["combined_text"] = (
        "장르: " + df["장르"] +
        " 감정: " + df["감정"] +
        " 평가: " + df["평가"]
    )
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

df = load_data()
model = load_model()

query = st.text_input("원하는 키워드나 감정을 입력하세요 (쉼표로 분리 가능)")

if query:
    if df.empty:
        st.warning("⚠ 먼저 작품을 하나 이상 입력해주세요.")
    else:
        query_list = [q.strip() for q in query.split(",")]
        query_embs = model.encode(query_list, convert_to_tensor=True)
        avg_query_emb = torch.mean(query_embs, dim=0, keepdim=True)

        doc_embs = model.encode(df["combined_text"].tolist(), convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(avg_query_emb, doc_embs)[0]
        sims = cos_scores.cpu().numpy()

        df["유사도"] = sims
        df_sorted = df.sort_values(by="유사도", ascending=False)

        # ✅ 알자르 타카르센 이미지 삽입
        st.image(
            "https://github.com/am0827/GBS/raw/d3b2b8f4beb4a2d3083e47bbb23afe971a1796fb/%EC%95%8C%EC%9E%90%EB%A5%B4_%ED%83%81%EC%B9%B4%EB%A5%B4%EC%84%BC.png",
            caption="이야기꾼 알자르 타카르센",
            width=200
        )
        st.markdown("🧙 이야기꾼 **알자르 타카르센**이 추천한 도서입니다.")

        st.write(f"🔍 알자르 타카르센의 추천 결과 상위 {min(5, len(df_sorted))}건")
        for _, row in df_sorted.head(5).iterrows():
            st.markdown(f"### {row['작품명']} — {row['저자']}")
            st.write(f"- 장르: {row['장르']} | 감정: {row['감정']}")
            st.write(f"- 평가: {row['평가']}")
            st.write(f"- **유사도**: {row['유사도']:.3f}")
            st.markdown("---")
