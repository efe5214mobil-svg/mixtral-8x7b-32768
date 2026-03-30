import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd
from PIL import Image

# 🎨 Sayfa Ayarları ve Koyu Tema (Siyahlımsı Gri)
st.set_page_config(page_title="MEB Yönetmelik Asistanı", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stChatMessage {
        background-color: #2d2d2d !important;
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] {
        background-color: #121212 !important;
    }
    .stTextInput input {
        background-color: #333 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()

# 🔑 API KEY
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🧠 VECTOR DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# --- SIDEBAR & SIRALAMA MANTIĞI ---
st.sidebar.header("📌 Sınıf Ders Programı")
dersprogram_klasor = "dersprogram_dosyasi"

siniflar = []
dosya_dict = {}

if os.path.exists(dersprogram_klasor):
    for dosya in os.listdir(dersprogram_klasor):
        if dosya.lower().endswith(".png"):
            # Orijinal temizleme mantığı
            sinif = dosya.replace(".png", "").upper().replace(" ", "")
            siniflar.append(sinif)
            dosya_dict[sinif] = os.path.join(dersprogram_klasor, dosya)
    
    # 🎯 Senin istediğin özel sıralama: Önce 12 -> 9, sonra A, B, C...
    def sinif_sort_key(s):
        numara_part = ''.join(filter(str.isdigit, s))
        numara = int(numara_part) if numara_part else 0
        harf = ''.join(filter(str.isalpha, s)) or ""
        return (-numara, harf) # Negatif numara büyükten küçüğe dizilmesini sağlar
    
    siniflar.sort(key=sinif_sort_key)
else:
    st.sidebar.warning(f"📂 Klasör bulunamadı: {dersprogram_klasor}")

# Seçim gösterim formatı: 9A -> 9 - A
def secim_gosterim_func(s):
    if len(s) >= 2 and s[0].isdigit():
        return f"{s[0:-1]} - {s[-1]}"
    return s

secim_gosterim = [secim_gosterim_func(s) for s in siniflar]

if siniflar:
    secim_index = st.sidebar.selectbox(
        "Sınıfı seçin:",
        range(len(secim_gosterim)),
        format_func=lambda x: secim_gosterim[x]
    )
    secim = siniflar[secim_index]
    img = Image.open(dosya_dict[secim])
    st.sidebar.image(img, caption=f"{secim_gosterim[secim_index]} Programı", use_container_width=True)

# ❌ GÜVENLİK FİLTRESİ
def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "siyaset", "din", "ırk", "ülke", "parti", "hükümet"]
    return any(kelime in soru.lower() for kelime in yasakli)

# 🤖 SORGULAMA (Detaycı & Açıklayıcı)
def okul_asistani_sorgula(soru):
    if uygunsuz_mu(soru):
        return "⚠️ Bu soru uygunsuz içerik barındırıyor. Sadece MEB yönetmeliği sorunuz.", None, None

    docs = vector_db.similarity_search(soru, k=4)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {"role": "system", "content": """Sen detaycı bir MEB yönetmelik uzmanısın. 
        Cevaplarına mutlaka EVET veya HAYIR ile başla. 
        Ardından yönetmelik maddelerine dayanarak detaylı ve teknik bir açıklama yap. 
        Sadece bağlama sadık kal, dışına çıkma."""}
    ]
    
    messages.extend(st.session_state.conversation[-2:])
    messages.append({"role": "user", "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=800
        )
        cevap = chat_completion.choices[0].message.content
    except:
        cevap = "Bir hata oluştu veya bu konuda bilgi bulunamadı."

    tablo_df = pd.DataFrame({"Madde Özeti": [doc.page_content[:150] + "..." for doc in docs]})
    kaynaklar = [doc.page_content for doc in docs]

    return cevap, tablo_df, kaynaklar

# --- CHAT ARAYÜZÜ ---
st.title("🎓 MEB Yönetmelik Asistanı")

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        cevap, tablo_df, kaynaklar = okul_asistani_sorgula(prompt)
        st.markdown(cevap)
        
        if tablo_df is not None:
            with st.expander("📊 Kaynak Maddeler"):
                st.table(tablo_df)
                for k in kaynaklar:
                    st.caption(k)
        
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
