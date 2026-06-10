import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import MobileNet

# ================= CONFIG =================

st.set_page_config(page_title="Fish Classifier", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "images.cv_jzk6llhf18tm3k0kyttxz",
    "data"
)

train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3
SAVED_MODEL_PATH = "best_fish_model.keras"

st.sidebar.title("🐟 Fish Classification App")
st.sidebar.info("Upload an image to classify the fish category.")

# ================= DATA =================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

class_names = list(train_data.class_indices.keys())

# ================= MODEL =================

def build_model():
    base_model = MobileNet(
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        weights="imagenet"
    )

    base_model.trainable = False

    model = Sequential([
        tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(len(class_names), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ================= TRAIN =================

@st.cache_resource
def train_model():
    st.write("🔧 Training MobileNet...")

    model = build_model()

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    model.save(SAVED_MODEL_PATH)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(history.history["accuracy"])
    ax[0].plot(history.history["val_accuracy"])
    ax[0].set_title("Accuracy")

    ax[1].plot(history.history["loss"])
    ax[1].plot(history.history["val_loss"])
    ax[1].set_title("Loss")

    st.pyplot(fig)

    return history.history["val_accuracy"][-1]

# ================= LOAD MODEL =================

@st.cache_resource
def load_trained_model():
    return load_model(SAVED_MODEL_PATH)

# ================= PREDICT =================

def predict_image(image_file):
    model = load_trained_model()

    image = load_img(image_file, target_size=IMAGE_SIZE)
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence_score = predictions[predicted_index] * 100

    return predicted_label, confidence_score, predictions

# ================= UI =================

tab1, tab2, tab3 = st.tabs(["📊 Train Model", "📷 Predict Fish",  "📈 Compare Models"])

with tab1:
    if st.button("🚀 Train Now"):
        accuracy = train_model()
        st.success(f"Training Complete! Validation Accuracy: {accuracy:.4f}")

with tab2:
    uploaded_image = st.file_uploader(
        "Upload Fish Image",
        type=["jpg", "jpeg", "png"]
    )

    with tab3:
        st.header("Model Comparison")

    if uploaded_image is not None:

        if not os.path.exists(SAVED_MODEL_PATH):
            st.warning("⚠ Please train the model first!")
        else:
            st.image(uploaded_image)

            label, conf_score, probabilities = predict_image(uploaded_image)

            st.success(f"Predicted: {label} ({conf_score:.2f}%)")

            st.bar_chart(
                pd.Series(probabilities, index=class_names)
            )

st.markdown("---")
st.markdown("✅ Multiclass Fish Image Classification Project")