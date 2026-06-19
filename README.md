# 🐟 Multiclass Fish Image Classification

A deep learning project that classifies fish images into multiple species categories using **transfer learning** (MobileNet) and serves predictions through an interactive **Streamlit** web app.

## Overview

This project trains a Convolutional Neural Network on a labeled fish image dataset and lets users upload a fish photo to get an instant species prediction along with a confidence score. The app also includes tabs for training the model and comparing model performance.

## Features

- **Image Classification** — Upload a fish image and get the predicted species with a confidence percentage.
- **Transfer Learning** — Uses a pre-trained **MobileNet** (ImageNet weights) as the feature extractor, with a custom classification head on top.
- **Data Augmentation** — Training images are augmented (zoom, rotation, horizontal flip) to improve generalization.
- **Interactive Dashboard** — Built with Streamlit, featuring:
  - 📊 **Train Model** tab — trigger model training
  - 📷 **Predict Fish** tab — upload an image and view predictions
  - 📈 **Compare Models** tab — view prediction probability breakdown as a bar chart
- **Cached Model Loading** — Uses Streamlit's resource caching so the trained model loads only once per session.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Base Model | MobileNet (pre-trained on ImageNet) |
| Web App | Streamlit |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |

## Project Structure

```
Multiclass-Fish-Image-Classification/
├── fish_image_classification_app.py   # Main Streamlit application
├── best_fish_model.keras              # Trained/saved Keras model
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Python runtime version (for deployment)
└── .python-version                    # Python version pin
```

> **Note:** The training script expects a `Dataset/images.cv_jzk6llhf18tm3k0kyttxz/data/` directory containing `train/`, `val/`, and `test/` subfolders organized by class. Make sure this dataset folder is present locally before training.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/karthick20022311/Multiclass-Fish-Image-Classification.git
   cd Multiclass-Fish-Image-Classification
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run fish_image_classification_app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

- Go to the **Train Model** tab to train the classifier on your dataset (if no saved model exists yet).
- Go to the **Predict Fish** tab to upload a fish image and view the predicted species and confidence score.
- Check the **Compare Models** tab to see the full probability distribution across all classes.

## Model Details

- **Input size:** 224×224 RGB images
- **Base architecture:** MobileNet (frozen base layers, `include_top=False`)
- **Classification head:** Global Average Pooling → Dense(256, ReLU) → Dropout(0.3) → Dense(softmax)
- **Optimizer:** Adam
- **Loss function:** Categorical Crossentropy

## Requirements

See [`requirements.txt`](requirements.txt):

```
streamlit==1.58.0
tensorflow-cpu==2.19.0
numpy
pandas
matplotlib
seaborn
pillow
scikit-learn
```

## License

This project is open-source and available for educational and personal use.

## Author

**Karthick** — [GitHub Profile](https://github.com/karthick20022311)
