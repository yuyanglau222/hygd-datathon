# 👁️ Glaucoma Detection AI: Datathon 2026

An AI-powered diagnostic tool designed to assist ophthalmologists in identifying Glaucoma from retinal fundus images. This project utilizes **MobileNetV2** for feature extraction and **Grad-CAM** for explainable AI (XAI), highlighting the specific regions of the eye influencing the model's decision.

## 📊 Project Overview

Glaucoma is a leading cause of irreversible blindness. Early detection is critical, but manual screening is time-consuming and prone to human error. This tool provides:

  * **High-Precision Classification:** Differentiates between Normal (GON-) and Glaucoma (GON+) scans.
  * **Visual Explanations:** Generates heatmaps to show where the AI is "looking" (e.g., the optic disc).
  * **Real-time Interface:** A Streamlit-based web app for instant diagnostic reporting.

## 📁 Repository Structure
```text
├── 01_model_training.py     # Data Preprocessing & Model Training
├── 02_model_evaluation.py   # Performance Metrics (ROC, Confusion Matrix)
├── 03_glaucoma_dashboard.py # Streamlit Web Interface (Main App)
├── Labels.csv               # Metadata & Clinical Labels
├── Images/                  # Dataset (700+ Retinal Fundus Images)
├── results/                 # Folder for model artifacts and plots
│   └── model_performance_summary.png  # Generated ROC & Confusion Matrix
├── LICENSE                  # MIT License
└── README.md                # Project Documentation & Guide
```

## 🚀 How to Run

### 1\. Requirements

Ensure you have Python installed, then install the dependencies:

```bash
pip install streamlit tensorflow opencv-python pillow pandas scikit-learn seaborn matplotlib
```

### 2\. Execution Pipeline

To reproduce the results, run the scripts in this specific order:

1.  **Train the Model:**

    ```bash
    python 01_model_training.py
    ```

    *This processes the `Images/` folder and saves the trained weights to the `results/` folder.*

2.  **Generate Metrics:**

    ```bash
    python 02_model_evaluation.py
    ```

    *This creates the ROC Curve and Confusion Matrix based on the trained model.*

3.  **Launch the App:**

    ```bash
    streamlit run 03_glaucoma_dashboard.py
    ```

## 📈 Model Performance

The model is evaluated based on its ability to minimize **False Negatives**, which is critical in a medical diagnostic context to ensure no cases of Glaucoma are missed.

  * **Architecture:** MobileNetV2 (Transfer Learning)
  * **Input Size:** 224x224 RGB
  * **Metrics:** View the performance summary below:

![Model Results](results/model_performance_summary.png)

## 🧠 Explainable AI (Grad-CAM)

We use **Gradient-weighted Class Activation Mapping (Grad-CAM)** to ensure clinical transparency. By visualizing the "Focus Map," doctors can verify if the AI is focusing on the optic nerve head rather than irrelevant background artifacts.

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

-----

### 💡 Note on Local Files

To maintain repository efficiency, the following generated files are excluded from the GitHub upload but will be created on your machine upon running the scripts:

  * `results/glaucoma_model.h5` (Trained Weights)
  * `results/images_resized/` (Preprocessed data)

