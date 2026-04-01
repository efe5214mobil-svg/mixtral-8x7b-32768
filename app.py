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

# 🖌️ Modern CSS - Arayüzü Şıklaştırma
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1rem; color: #1E1E1E; }
    
    /* 🚀 Sağ Altta Yüzen Buton Tasarımı */
    .floating-button-container {
        position: fixed;
        bottom: 90px; 
        right: 5%; 
        z-index: 999999;
    }

    .stLinkButton a {
        background-color: #FF8C00 !important;
        color: white !important;
        border-radius: 30px !important;
        padding: 0.7rem 1.8rem !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25) !important;
        transition: all 0.3s ease;
    }
    
    .stLinkButton a:hover {
        background-color: #E67E22 !important;
        transform: scale(1.1);
    }

    /* 💡 Öneri Kartları Tasarımı */
    .category-box {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 20px;
        border-top: 5px solid #FF4B4B;
        height: 100%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: 0.3s ease;
    }
    .category-box:hover {
        background-color: rgba(255, 75, 75, 0.08);
        transform: translateY(-3px);
    }
    .category-title { font-weight: 800; color: #FF4B4B; margin-bottom: 12px; font-size: 1.2rem; }
    .category-item { font-size: 0.9rem; margin-bottom: 8px; color: #333; line-height: 1.4; }
    
    @media (prefers-color-scheme: dark) {
        .category-item { color: #DDD; }
        .category-box { background-color: rgba(255, 255, 255, 0.05); }
        .main-title { color: #FFFFFF; }
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Vektör Veritabanı Yükleme
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

# 🛡️ Güvenlik ve İçerik Filtresi
def icerik_denetimi(soru):
    yasakli = ["siyaset", "parti", "seçim", "din", "mezhep", "ırk", "küfür", "argo", "hakaret", "aptal", "salak"]
    soru_temiz = soru.lower()
    return any(kelime in soru_temiz for kelime in yasakli)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🤖 Ana Sorgulama Motoru
def sorgula(soru):
    # 1. Filtre Kontrolü
    if icerik_denetimi(soru):
        return "⚠️ Üzgünüm, topluluk kuralları gereği siyaset, din, ırk veya hakaret içeren mesajlara yanıt veremiyorum. Size sadece MEB yönetmeliğiyle ilgili konularda yardımcı olabilirim. [TABLO_YOK]", []

    # 2. Benzerlik Araması
    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [{
        "role": "system", 
        "content": """Sen MEB yönetmelik uzmanısın. 
        - Selamlaşmalara (merhaba, selam vb.) dostane ama kısa cevap ver.
        - Mevzuat dışı konularda veya selamlaşmalarda cevabın sonuna [TABLO_YOK] ekle.
        
        [PDF KRİTİK VERİLERİ]:
        - Devamsızlık: Özürsüz 10 gün, Toplam 30 gün sınır. (Bazı durumlarda 60 gün)
        - Başarı: Sınıf geçme notu 50.00.
        - Sınıf Geçme: En fazla 3 dersten sorumlu geçilebilir. 6+ zayıf doğrudan sınıf tekrarıdır.
        - Ödül Puanları: Teşekkür belgesi (70.00 - 84.99), Takdir belgesi (85.00 - 100).
        - Disiplin: Kopya ve sigara kullanımı 'Kınama' cezasıdır.
        - Kayıt: Evli öğrencilerin kayıtları silinir/yapılamaz."""
    }]
    
    # Hafıza: Son 4 mesajı dahil et
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": f"MEVZUAT BAĞLAMI:\n{baglam}\n\nKULLANICI SORUSU: {soru}"})
    
    completion = client.chat.completions.create(
        messages=messages, 
        model="llama-3.3-70b-versatile", 
        temperature=0.2
    )
    return completion.choices[0].message.content, docs

# --- ARAYÜZ KATMANI ---

st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

# 🚀 Yüzen Sınıf Programı Butonu
st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 Öneri Kartları (Dashboard)
if not st.session_state.conversation:
    st.markdown("### 💡 Başlamanız İçin Bazı Öneriler")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="category-box"><div class="category-title">📜 Disiplin</div><div class="category-item">• Kopya çekmenin cezası nedir?<br>• Kınama cezası hangi durumlarda verilir?</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">• Kaç gün devamsızlık hakkım var?<br>• 30 gün kuralı ne demek?</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı</div><div class="category-item">• Takdir almak için kaç puan gerekir?<br>• Kaç zayıfla sınıfta kalınır?</div></div>', unsafe_allow_html=True)
    st.markdown("<br><hr>", unsafe_allow_html=True)

# 💬 Mesajları Görüntüle
for mesaj in st.session_state.conversation:
    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"].replace("[TABLO_YOK]", ""))

# ⌨️ Kullanıcı Girişi
if soru_input := st.chat_input("Yönetmelik veya okul kuralları hakkında bir soru yazın..."):
    st.session_state.conversation.append({"role": "user", "content": soru_input})
    with st.chat_message("user"):
        st.markdown(soru_input)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Mevzuat dosyaları taranıyor..."):
            cevap, kaynaklar = sorgula(soru_input)
            
            tablo_gizle = "[TABLO_YOK]" in cevap
            temiz_cevap = cevap.replace("[TABLO_YOK]", "").strip()
            
            st.markdown(temiz_cevap)
            
            # Kaynak Tablosu (Sadece mevzuat sorularında ve Konum sütunu olmadan)
            if kaynaklar and not tablo_gizle:
                st.markdown("---")
                st.markdown("📑 **İlgili Mevzuat Referansları**")
                
                tablo_verisi = []
                for index, doc in enumerate(kaynaklar):
                    tablo_verisi.append({
                        "Kaynak": f"Madde {index + 1}",
                        "İçerik Özeti": doc.page_content[:250] + "..."
                    })
                
                # Pandas DataFrame ile şık bir tablo basımı
                df = pd.DataFrame(tablo_verisi)
                st.table(df)
            
            st.session_state.conversation.append({"role": "assistant", "content": temiz_cevap})
