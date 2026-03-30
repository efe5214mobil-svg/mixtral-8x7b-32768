import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd
from PIL import Image

# 🎨 Gelişmiş Gradyanlı Dark Tema
st.set_page_config(page_title="MEB Yönetmelik Asistanı", layout="wide")

st.markdown("""
    <style>
    /* Arka plan gradyanı */
    .stApp {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d3436 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar düzeni */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 15, 15, 0.8) !important;
        border-right: 1px solid #444;
    }
    
    /* Mesaj balonları */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px !important;
        margin-bottom: 10px;
    }

    /* Input alanı */
    .stTextInput input {
        background-color: #252525 !important;
        color: white !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# --- SIDEBAR & DOSYA İSMİ TEMİZLEME ---
st.sidebar.header("📌 Sınıf Panosu")
dersprogram_klasor = "dersprogram_dosyasi"

dosya_haritasi = {} # { "Görünen Ad": "Gerçek Yol" }

if os.path.exists(dersprogram_klasor):
    dosyalar = [f for f in os.listdir(dersprogram_klasor) if f.lower().endswith(".png")]
    
    # 🎯 İsim Temizleme Mantığı (12A.PN-G -> 12 - A)
    for d in dosyalar:
        # Sadece rakamları ve harfleri ayıkla (Örn: 12A)
        temiz_isim = re.sub(r'[^a-zA-Z0-9]', '', d.replace(".png", ""))
        
        # Formatlama: "12A" -> "12 - A"
        match = re.match(r"(\d+)([a-zA-Z]+)", temiz_isim)
        if match:
            gosterim_adi = f"{match.group(1)} - {match.group(2).upper()}"
        else:
            gosterim_adi = temiz_isim.upper()
            
        dosya_haritasi[gosterim_adi] = os.path.join(dersprogram_klasor, d)

    # Sıralama (Önce 12, sonra 11...)
    sirali_isimler = sorted(
        dosya_haritasi.keys(), 
        key=lambda x: (-int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0, x)
    )

    if sirali_isimler:
        secilen_sinif = st.sidebar.selectbox("Sınıfı seçin:", sirali_isimler)
        st.sidebar.image(Image.open(dosya_haritasi[secilen_sinif]), caption=f"{secilen_sinif} Ders Programı")
else:
    st.sidebar.error("Klasör bulunamadı!")

# ❌ Güvenlik
def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "siyaset", "din", "hakaret"]
    return any(k in soru.lower() for k in yasakli)

# 🤖 Sorgulama
def okul_asistani_sorgula(soru):
    if uygunsuz_mu(soru):
        return "⚠️ Bu içerik kurallarımıza uygun değil.", None

    docs = vector_db.similarity_search(soru, k=4)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {"role": "system", "content": "Sen detaycı bir MEB uzmanısın. Cevabına EVET/HAYIR ile başla ve madde madde açıkla."}
    ]
    messages.extend(st.session_state.conversation[-2:])
    messages.append({"role": "user", "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"})

    completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1000
    )
    return completion.choices[0].message.content, docs

# --- ANA EKRAN ---
st.title("🎓 MEB Yönetmelik Asistanı")

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Soru sorun..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        cevap, kaynaklar = okul_asistani_sorgula(prompt)
        st.markdown(cevap)
        
        if kaynaklar:
            with st.expander("📚 Kaynak Yönetmelik Metni"):
                for k in kaynaklar:
                    st.caption(k.page_content)
        
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
