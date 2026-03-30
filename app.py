import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re

# 🔐 .env yükle
load_dotenv()

# 🔑 API KEY
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

# 🎯 Başlık
st.title("MEB Yönetmelik Asistanı - Sohbet Hafızalı")

# 🧠 VECTOR DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="okul_asistani_gpt_db",
        embedding_function=embeddings
    )
    return db

vector_db = load_vector_db()

# 🗂️ Session State ile sohbet geçmişini tut
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🔹 Uygunsuz soruları engelle
def uygunsuz_soru_kontrol(soru):
    yasak_kelimeler = [
        "eşcinsel", "uyuşturucu", "küfür", "hakaret", "cinsel", "porno"
    ]
    for kelime in yasak_kelimeler:
        if kelime.lower() in soru.lower():
            return True
    return False

# 🔹 Baglam temizleme
def temizle(text):
    text = re.sub(r'(\d{4}/\s*){2,}', '', text)  # tekrar eden 2005/ gibi şeyleri sil
    text = re.sub(r'\s+', ' ', text)             # fazla boşlukları tek boşluk yap
    return text.strip()

# 🤖 SORGULAMA
def okul_asistani_sorgula(soru):

    # 🔴 uygunsuz soruları engelle
    if uygunsuz_soru_kontrol(soru):
        return "Üzgünüm, bu tür sorulara cevap veremem."

    # 🔍 arama sorgusu
    arama_sorgusu = f"{soru} meb yönetmelik maddesi devamsızlık şartları"

    docs = vector_db.similarity_search_with_score(arama_sorgusu, k=5)
    docs = sorted(docs, key=lambda x: x[1])[:3]
    docs = [doc[0] for doc in docs]

    if not docs:
        return "Veri bulunamadı."

    # baglam temizle
    baglam = "\n\n".join([temizle(doc.page_content[:500]) for doc in docs])

    # mesajlar
    messages = [
        {"role": "system", "content": """
Sen MEB yönetmeliği uzmanısın.
Kurallar:
- Sadece verilen bağlama göre cevap ver
- Eğer madde varsa belirt
- Gereksiz tekrar yapma
- Uyduruk veya uygunsuz sorulara cevap verme
"""}
    ]

    # önceki sohbeti ekle
    for msg in st.session_state.conversation:
        messages.append(msg)

    # kullanıcı mesajını en sonda ekle
    messages.append({"role": "user", "content": f"{baglam}\n\nSoru: {soru}"})

    # chat completion
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=500
    )

    cevap = chat_completion.choices[0].message.content

    # session state güncelle
    st.session_state.conversation.append({"role": "user", "content": soru})
    st.session_state.conversation.append({"role": "assistant", "content": cevap})

    # kaynak ekleme
    kaynaklar = [doc.page_content[:200] for doc in docs]

    # kullanıcı mesajını en sonda göster
    final_output = cevap + "\n\n📚 Kaynak:\n- " + "\n- ".join(kaynaklar) + f"\n\n💬 Siz: {soru}"
    return final_output

# 💬 Chat Arayüzü
for msg in st.session_state.conversation:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın:"):
    with st.spinner("Yanıt hazırlanıyor..."):
        cevap = okul_asistani_sorgula(prompt)
        with st.chat_message("assistant"):
            st.markdown(cevap)
