import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# 🛠 Ayarlar
st.set_page_config(page_title="Tüberküloz Teşhis Sistemi", page_icon="🦠")
st.title("🦠 Tüberküloz Teşhis Sistemi")

# 🤖 Model yükleme (Önbelleğe alıyoruz ki her seferinde yüklemesin)
@st.cache_resource
def load_my_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "tb_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_my_model()

# 🔍 Tahmin fonksiyonu
def predict(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    label = "🦠 Tüberküloz Var" if pred > 0.5 else "✅ Tüberküloz Yok"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, round(confidence * 100, 2)

# 📤 Dosya Yükleme Alanı
uploaded_file = st.file_uploader("Röntgen filmini yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Görüntü', use_container_width=True)
    
    if st.button("Tahmin Et"):
        label, confidence = predict(img)
        
        if "Var" in label:
            st.error(f"Sonuç: {label} (Güven Oranı: %{confidence})")
        else:
            st.success(f"Sonuç: {label} (Güven Oranı: %{confidence})")
