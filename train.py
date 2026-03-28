import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 📁 Ana dizin
base_dir = os.path.dirname(os.path.abspath(__file__))

# 📁 Klasörler
model_dir = os.path.join(base_dir, "model")
static_dir = os.path.join(base_dir, "static")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

# 📁 Dataset yolu
data_dir = os.path.join(base_dir, "dataset")

# 📊 Veri artırma (overfitting azaltır)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

# 🎯 Eğitim verisi
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

# 🎯 Validation verisi
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# 🔥 CNN tabanlı hazır model (çok güçlü)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# ❌ ağırlıkları dondur (transfer learning)
base_model.trainable = False

# 🔗 üst katmanlar
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

# 🧠 model oluştur
model = models.Model(inputs=base_model.input, outputs=output)

# ⚙️ compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🚀 eğitim
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# 💾 model kaydet
model_path = os.path.join(model_dir, "tb_model.h5")
model.save(model_path)

print("MODEL KAYDEDİLDİ:", model_path)

# =========================
# 📊 GRAFİKLER
# =========================

# 📈 Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Validation")
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(static_dir, "accuracy.png"))
plt.close()

# 📉 Loss
plt.figure()
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Validation")
plt.legend()
plt.title("Loss")
plt.savefig(os.path.join(static_dir, "loss.png"))
plt.close()

# 📊 Confusion Matrix
val_data.reset()
preds = model.predict(val_data)
preds = (preds > 0.5).astype(int).reshape(-1)

cm = confusion_matrix(val_data.classes, preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(static_dir, "confusion_matrix.png"))
plt.close()

print("GRAFİKLER OLUŞTU ✅")