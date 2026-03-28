import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# 1. Sayfa Ayarları
st.set_page_config(page_title="Tüberküloz Teşhis Sistemi", page_icon="🦠", layout="centered")

# 2. Başlık ve Açıklama
st.title("🦠 Tüberküloz Teşhis Sistemi")
st.write("Yapay zeka destekli röntgen analizi. Lütfen bir göğüs röntgeni yükleyin.")

# 3. Model Yükleme (Hız için önbelleğe alınır)
@st.cache_resource
def load_my_model():
    # Görsellerinizde modelin ana dizinde olduğu görülüyor
    model_path = "tb_model.h5"
    if not os.path.exists(model_path):
        st.error(f"HATA: '{model_path}' dosyası bulunamadı! Lütfen dosyanın GitHub ana dizininde olduğundan emin olun.")
        return None
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# 4. Dosya Yükleme Alanı
uploaded_file = st.file_uploader("Röntgen filmi seçin (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Resmi görüntüle
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Yüklenen Röntgen Görüntüsü', use_container_width=True)
    
    # "Analiz Et" butonu
    if st.button("Hemen Analiz Et"):
        with st.spinner('Yapay zeka görüntüyü inceliyor...'):
            try:
                # Görüntü ön işleme (Modelin eğitildiği boyutlara getirilir)
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = img_array / 255.0  # Normalize etme
                img_array = np.expand_dims(img_array, axis=0) # Batch boyutu ekleme
                
                # Tahmin yapma
                prediction = model.predict(img_array)[0][0]
                
                # Sonuçları ekrana yazdırma
                st.divider()
                if prediction > 0.5:
                    st.error(f"### SONUÇ: Tüberküloz Belirtisi Saptandı 🦠")
                    confidence = prediction * 100
                else:
                    st.success(f"### SONUÇ: Sağlıklı / Belirti Saptanmadı ✅")
                    confidence = (1 - prediction) * 100
                
                st.info(f"**Yapay Zeka Güven Oranı:** %{round(float(confidence), 2)}")
                
            except Exception as e:
                st.error(f"Analiz sırasında bir hata oluştu: {e}")

# Alt bilgi
st.caption("Not: Bu bir yapay zeka projesidir, kesin tıbbi sonuç yerine geçmez.")
