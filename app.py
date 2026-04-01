import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import pandas as pd

# 🔐 API ve Çevre Değişkenleri
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🎨 Sayfa Yapılandırması
st.set_page_config(page_title="MEB Mevzuat Asistanı", page_icon="🏛️", layout="centered")

# 🖌️ Modern CSS
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; }
    
    /* 🚀 Sağ Altta Yüzen Buton */
    .floating-button-container {
        position: fixed;
        bottom: 85px; 
        right: 10%; 
        z-index: 999999;
    }

    .stLinkButton a {
        background-color: #FF8C00 !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        border: 2px solid white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }

    /* 💡 Öneri Kartları Stili */
    .category-box {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
        transition: 0.3s ease;
    }
    .category-box:hover { transform: translateY(-5px); background-color: rgba(255, 75, 75, 0.05); }
    .category-title { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .category-item { font-size: 0.88rem; margin-bottom: 6px; color: #444; }
    
    @media (prefers-color-scheme: dark) {
        .category-item { color: #CCC; }
        .category-box { background-color: rgba(255, 255, 255, 0.05); }
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Vektör Veritabanı
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

# 🛡️ Güvenlik Filtresi (Küfür, Din, Dil, Irk, Siyaset)
def guvenlik_filtresi(metin):
    yasakli_kelimeler = [
        "küfür_kelimesi1", "argo1", "siyaset", "parti", "seçim", 
        "din", "mezhep", "ırk", "etnik", "hakaret", "aptal", "salak"
    ]
    metin_lower = metin.lower()
    for kelime in yasakli_kelimeler:
        if kelime in metin_lower:
            return True
    return False

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🤖 Sorgulama Fonksiyonu
def sorgula(soru):
    # 1. Filtre Kontrolü
    if guvenlik_filtresi(soru):
        return "⚠️ Üzgünüm, topluluk kuralları gereği küfür, argo, din, siyaset veya ırkçılık içeren ifadelere yanıt veremiyorum. Size sadece MEB yönetmeliği konusunda yardımcı olabilirim. [TABLO_YOK]", []

    # 2. Vektör Arama
    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [{
        "role": "system", 
        "content": """Sen MEB yönetmelik uzmanısın. 
        - Selamlaşmalara samimi cevap ver ama tablo ekleme.
        - Mevzuat dışı sorularda cevabın sonuna [TABLO_YOK] ekle.
        
        [SABİT PDF VERİLERİ]:
        - Devamsızlık: Özürsüz 10, Toplam 30 gün.
        - Başarı Notu: Geçme sınırı 50.
        - Sınıf Geçme: En fazla 3 ders sorumluluk, 6+ zayıf sınıf tekrarı.
        - Ödül: Teşekkür 70+, Takdir 85+.
        - Disiplin: Kopya/Sigara 'Kınama' gerektirir.
        - Kayıt: Evli öğrencilerin ilişiği kesilir."""
    }]
    
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})
    
    completion = client.chat.completions.create(
        messages=messages, 
        model="llama-3.3-70b-versatile", 
        temperature=0.2
    )
    return completion.choices[0].message.content, docs

# --- ARAYÜZ ---
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

# 🚀 Sağ Alt Buton
st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 Öneri Kartları (Sohbet boşken)
if not st.session_state.conversation:
    st.markdown("### 💡 Hızlı Sorular")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="category-box"><div class="category-title">📜 Disiplin</div><div class="category-item">"Kopya çekmenin cezası nedir?"</div><div class="category-item">"Hangi davranışlar kınama alır?"</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">"Kaç gün devamsızlık hakkım var?"</div><div class="category-item">"Özürsüz devamsızlık sınırı kaç?"</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı</div><div class="category-item">"Takdir almak için kaç puan gerekir?"</div><div class="category-item">"Sınıfta kalma kuralı nedir?"</div></div>', unsafe_allow_html=True)
    st.markdown("<br><hr>", unsafe_allow_html=True)

# 💬 Sohbet Akışı
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("[TABLO_YOK]", ""))

# ⌨️ Giriş Alanı
if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ İnceleniyor..."):
            cevap, kaynaklar = sorgula(prompt)
            
            tablo_gizle = "[TABLO_YOK]" in cevap
            temiz_cevap = cevap.replace("[TABLO_YOK]", "").strip()
            
            st.markdown(temiz_cevap)
            
            if kaynaklar and not tablo_gizle:
                st.markdown("---")
                st.markdown("📑 **Referans Maddeler**")
                
                ref_data = []
                for i, doc in enumerate(kaynaklar):
                    ref_data.append({
                        "Kaynak": f"Madde {i+1}",
                        "İçerik Özeti": doc.page_content[:250] + "..."
                    })
                
                st.table(pd.DataFrame(ref_data))
            
            st.session_state.conversation.append({"role": "assistant", "content": temiz_cevap})
