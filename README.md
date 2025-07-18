# 🧠 Brain Tumor Classification using CNN

This project classifies brain MRI images into four categories using a Convolutional Neural Network (CNN):
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

A web app built with Streamlit allows users to upload images and get instant predictions.

---

## 📁 Dataset

Dataset used: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Folder structure:
```
dataset/
├── glioma_tumor/
├── meningioma_tumor/
├── pituitary_tumor/
└── no_tumor/
```

---

## 🧱 Project Structure
```
BrainMRI-Images/
├── dataset/                   # Contains MRI images
├── models/
│   └── brain_tumor_model.h5  # Trained model
├── notebook/
│   └── train_model.ipynb     # Training notebook
├── brain_tumor_app.py        # Streamlit web app
├── requirements.txt          # Project dependencies
└── README.md                 # Project description
```

---

## ⚙️ Technologies
- Python 3.10+
- TensorFlow / Keras
- Streamlit
- NumPy, Pandas, Pillow
- Matplotlib, Seaborn
- Scikit-learn

---

## 🚀 How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Train model (optional)
Run `notebook/train_model.ipynb` in Colab or Jupyter to generate `brain_tumor_model.h5`.

### 3. Launch the app
```bash
streamlit run brain_tumor_app.py
```

---

## 📈 Example Output
- Accuracy ~90–95% across 4 classes
- Displays prediction with confidence score

---

## 📜 License
This project is for educational and research purposes only.