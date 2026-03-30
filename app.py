import streamlit as st
from rag import okul_asistani_sorgula
from vector_store import load_vector_db
from groq import Groq

# 🔐 API KEY
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# 🎨 SAYFA AYARI
st.set_page_config(page_title="Okul Asistanı", page_icon="🎓")

st.title("🎓 MEB Yönetmelik Asistanı")

# 🧠 VECTOR DB
@st.cache_resource
def get_db():
    return load_vector_db()

vector_db = get_db()

# 🧠 HAFIZA
if "messages" not in st.session_state:
    st.session_state.messages = []

# 💬 ESKİ MESAJLAR
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ✍️ KULLANICI GİRİŞİ
soru = st.chat_input("Sorunuzu yazın")

if soru:
    # kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": soru})

    with st.chat_message("user"):
        st.write(soru)

    # AI cevabı
    with st.chat_message("assistant"):
        with st.spinner("Cevap hazırlanıyor..."):
            cevap, kaynaklar = okul_asistani_sorgula(soru, vector_db)

        st.write(cevap)

        # 📚 kaynaklar
        with st.expander("📚 Kaynaklar"):
            for k in kaynaklar:
                st.markdown(f"- {k}...")

    # hafızaya ekle
    st.session_state.messages.append({
        "role": "assistant",
        "content": cevap
    })
