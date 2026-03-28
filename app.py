import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# Başlık
st.title("🦠 Tüberküloz Teşhis Sistemi")

# Model Yükleme
@st.cache_resource
def load_my_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "tb_model.h5")
    return tf.keras.models.load_model(model_path)

try:
    model = load_my_model()
    st.success("Model başarıyla yüklendi!")
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")

# Dosya Yükleme
uploaded_file = st.file_uploader("Röntgen filmi seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Yüklenen Görüntü', use_container_width=True)
    
    # Tahmin
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)[0][0]
    label = "🦠 Tüberküloz Var" if pred > 0.5 else "✅ Tüberküloz Yok"
    st.subheader(f"Sonuç: {label}")
    st.write(f"Güven Oranı: %{round(float(pred if pred > 0.5 else 1-pred)*100, 2)}")
