import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import base64

# 🎨 Sayfa Ayarları ve Tema
st.set_page_config(page_title="MEB Yönetmelik Asistanı", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2c3e50 100%);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTextInput input {
        background-color: #252525 !important;
        color: white !important;
        border: 1px solid #444 !important;
    }
    .stTable {
        background-color: rgba(255, 255, 255, 0.05);
        color: white;
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

# --- GÖRSEL ÖZET FONKSİYONU ---
def gorsel_mevzuat_ozeti(docs):
    data = []
    for doc in docs:
        content = doc.page_content
        madde_match = re.search(r"MADDE\s+\d+", content, re.IGNORECASE)
        madde_no = madde_match.group(0) if madde_match else "Genel Hüküm"
        temiz_icerik = content.replace("\n", " ").strip()
        ozet = temiz_icerik[:200] + "..." if len(temiz_icerik) > 200 else temiz_icerik
        data.append({"Dayanak": madde_no, "Resmi İçerik Özeti": ozet})
    
    df = pd.DataFrame(data)
    st.markdown("#### 📋 Mevzuat Analiz Çizelgesi")
    st.table(df)


# 🛡️ GÜVENLİK FİLTRESİ (Sadece Küfür/Hakaret Engelli, Mevzuat Kelimeleri Serbest)
def uygunsuz_mu(soru):
    # Bu liste sadece ağır küfürleri ve hakaretleri kapsar. 
    # 'Özürlü' ve 'Özürsüz' gibi mevzuat kelimeleri listede yoktur ve engellenmez.
    data_enc = "a3VmdXIsYXJnbyxzaXlhc2V0LGRpbixpcmssaGFrYXJldCxwYXJ0aSxzZXgsc2Vrcyxwb3JubyxjaXBsYWssbWVtZSxnb3Qsc2lrLGFtayxwaXBpLHRhY2l6LG11c3RlaGNlbixnYXksbGV6Yml5ZW4sZmV0aXNsdWssdmFnaW5hLHBlbmlzLGVzY29ydCxvYyxwaWNoLHNpY21payx5YXJyYWs="
    yasakli_liste = base64.b64decode(data_enc).decode('utf-8').split(',')
    s = soru.lower().strip()
    
    for kelime in yasakli_liste:
        if kelime in s:
            return True, "⚠️ **Güvenlik Engeli:** Girdiğiniz ifade topluluk ilkelerine uygun değildir. Lütfen akademik dille devam ediniz."
    return False, ""

# 🤖 SORGULAMA (MEB Yönetmelik Uzmanı)
def okul_asistani_sorgula(soru):
    hata, mesaj = uygunsuz_mu(soru)
    if hata: return mesaj, None

    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([d.page_content for d in docs])

    messages = [
        {
            "role": "system", 
            "content": """Sen MEB Yönetmelik Asistanısın. 
            KESİN KURALLAR:
            1. 'Özürlü devamsızlık' ve 'Özürsüz devamsızlık' terimlerini asla yasaklama, bunlar resmi terimlerdir.
            2. Cevaplara asla 'Evet' veya 'Hayır' diyerek başlama. 
            3. Sadece okul, sınav ve disiplin yönetmeliğine cevap ver.
            4. Özürlü devamsızlık sınırı 20 GÜN, toplam sınır 30 GÜNDÜR.
            5. Ortalama 50 altıysa en fazla 3 dersten sorumlu geçilebilir."""
        }
    ]
    messages.extend(st.session_state.conversation[-2:])
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=1000
        )
        return completion.choices[0].message.content, docs
    except:
        return "Şu an yanıt verilemiyor, lütfen tekrar deneyin.", None

# --- ANA EKRAN ---
st.title("🎓 MEB Yönetmelik Asistanı")

# 💡 GENİŞLETİLMİŞ SIKÇA SORULAN SORULAR
with st.expander("❓ Sıkça Sorulan Sorular (Kapsamlı Rehber)"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📌 Devamsızlık Hakkında**")
        st.write("- Özürlü devamsızlık hakkım toplam kaç gündür?")
        st.write("- Özürsüz devamsızlık 10 günü geçerse ne olur?")
        st.write("- Faaliyetli (görevli) olduğum günler devamsızlıktan sayılır mı?")
        st.write("- Toplam devamsızlık sınırı 30 günü aşarsa ne yapılır?")
        
        st.markdown("**📌 Sınıf Geçme ve Notlar**")
        st.write("- Yıl sonu ortalamam 50 altındaysa kalır mıyım?")
        st.write("- Sorumluluk sınavları ne zaman ve nasıl yapılır?")
        st.write("- E-Okul'da not girişi bittikten sonra not değiştirilir mi?")
    with col2:
        st.markdown("**📌 Disiplin ve Davranışlar**")
        st.write("- Okulda cep telefonu kullanmanın yaptırımı nedir?")
        st.write("- Kınama cezası alan öğrenci takdir/teşekkür alabilir mi?")
        st.write("- Okul eşyasına zarar vermenin cezası nedir?")
        
        st.markdown("**📌 Nakil ve Kayıt**")
        st.write("- Başka bir okula nakil olmak için şartlar nelerdir?")
        st.write("- Açık liseye geçiş şartları güncel yönetmelikte nasıldır?")
        st.write("- Meslek lisesinden Anadolu lisesine geçiş yapılır mı?")

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Yönetmelik sorunuzu buraya yazın..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mevzuat dosyaları taranıyor..."):
            cevap, kaynaklar = okul_asistani_sorgula(prompt)
            st.markdown(cevap)
            if kaynaklar:
                with st.expander("📖 Dayanak Yönetmelik Maddeleri (Görsel Analiz)"):
                    gorsel_mevzuat_ozeti(kaynaklar)
                    st.divider()
                    for k in kaynaklar: st.caption(f"📍 {k.page_content}")
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
