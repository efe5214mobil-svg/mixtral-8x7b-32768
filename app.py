import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🎯 Başlık
st.title("🎓 MEB Yönetmelik Asistanı")

# 🔐 API KEY
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# 🧠 VECTOR DB YÜKLE (EN KRİTİK KISIM)
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory="okul_asistani_gpt_db",
        embedding_function=embeddings
    )
    return db

vector_db = load_vector_db()

# 🤖 SORGULAMA
def okul_asistani_sorgula(soru):
    arama_sorgusu = f"{soru} yönetmelik maddesi şartları ve sınırları"
    docs = vector_db.similarity_search(arama_sorgusu, k=5)

    baglam = "\n\n".join([doc.page_content for doc in docs])

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Sen MEB yönetmeliği uzmanısın. Cevabını sadece verilen bağlama göre ver."
            },
            {
                "role": "user",
                "content": f"Bağlam: {baglam}\n\nSoru: {soru}"
            }
        ],
        model="llama3-8b-8192",
        temperature=0,
    )

    return chat_completion.choices[0].message.content

# 💬 ARAYÜZ
soru = st.text_input("Sorunuzu yazın:")

if soru:
    with st.spinner("Yanıt hazırlanıyor..."):
        cevap = okul_asistani_sorgula(soru)
        st.success(cevap)
