import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# Sayfa Yapılandırması
st.set_page_config(page_title="Tüberküloz Teşhis", page_icon="🦠")
st.title("🦠 Tüberküloz Teşhis Sistemi")

# 🤖 Model yükleme (Önbelleğe alarak hızı artırıyoruz)
@st.cache_resource
def load_my_model():
    # Model dosyan ana dizinde olduğu için direkt adıyla çağırıyoruz
    model_path = "tb_model.h5" 
    return tf.keras.models.load_model(model_path)

try:
    model = load_my_model()
    st.sidebar.success("Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenemedi! 'tb_model.h5' dosyasının GitHub ana dizininde olduğundan emin olun. Hata: {e}")

# 📂 Dosya Yükleme Alanı
uploaded_file = st.file_uploader("Röntgen filmi seçin (JPG, PNG, JPEG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Resmi ekranda göster
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Görüntü', use_container_width=True)
    
    if st.button("Analiz Et"):
        with st.spinner('Yapay zeka analiz ediyor...'):
            # Resim ön işleme
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Tahmin
            pred = model.predict(img_array)[0][0]
            
            # Sonuç gösterimi
            if pred > 0.5:
                st.error(f"Sonuç: 🦠 Tüberküloz Belirtisi Saptandı")
                st.write(f"Güven Oranı: %{round(float(pred) * 100, 2)}")
            else:
                st.success(f"Sonuç: ✅ Sağlıklı / Belirti Saptanmadı")
                st.write(f"Güven Oranı: %{round(float(1 - pred) * 100, 2)}")
