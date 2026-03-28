from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# 📁 Ana dizin
base_dir = os.path.dirname(os.path.abspath(__file__))

# 🤖 Model yükleme
model = tf.keras.models.load_model(os.path.join(base_dir, "model", "tb_model.h5"))

# 🔍 Tahmin fonksiyonu
def predict(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    label = "🦠 Tüberküloz Var" if pred > 0.5 else "✅ Tüberküloz Yok"
    confidence = pred if pred > 0.5 else 1 - pred

    return label, round(confidence * 100, 2)

# 🌐 Ana sayfa
@app.route("/", methods=["GET","POST"])
def index():
    result = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename.replace(" ", "_")

        static_path = os.path.join(base_dir, "static")
        os.makedirs(static_path, exist_ok=True)

        path = os.path.join(static_path, filename)
        file.save(path)

        result, confidence = predict(path)
        img_path = filename

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path
    )

# 📄 Kodları gösterme sayfası
@app.route("/code")
def show_code():
    with open(os.path.join(base_dir, "train.py"), "r", encoding="utf-8") as f:
        train_code = f.read()

    with open(os.path.join(base_dir, "app.py"), "r", encoding="utf-8") as f:
        app_code = f.read()

    return render_template("code.html", train_code=train_code, app_code=app_code)

if __name__ == "__main__":
    app.run(debug=True)