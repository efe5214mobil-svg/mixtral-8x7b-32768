import streamlit as st
import os
from groq import Groq

# 1. Başlık ve Arayüz
st.title("MEB Yönetmelik Asistanı")

# 2. Secrets (Gizli Anahtarlar) Yönetimi
# Streamlit Cloud'da 'Secrets' kısmına GROQ_API_KEY eklemelisin
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# NOT: vector_db nesnesini burada tanımlaman veya yüklemen gerekecek.
# Eğer veritabanın bir dosya ise (FAISS vb.), onu da Github'a yüklemelisin.

def okul_asistani_sorgula(soru):
    # vector_db'nin burada erişilebilir olduğunu varsayıyoruz
    arama_sorgusu = f"{soru} yönetmelik maddesi şartları ve sınırları"
    docs = vector_db.similarity_search(arama_sorgusu, k=10)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """Sen MEB Ortaöğretim Kurumları Yönetmeliği uzmanısın..."""
            },
            {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {soru}"}
        ],
        model="llama3-8b-8192", # Groq üzerinde geçerli bir model ismi kullandığından emin ol
        temperature=0,
    )
    return chat_completion.choices[0].message.content

# 3. Kullanıcı Girişi
soru = st.text_input("Sormak istediğiniz konuyu yazın:")
if soru:
    with st.spinner("Yanıt hazırlanıyor..."):
        cevap = okul_asistani_sorgula(soru)
        st.write(cevap)
