import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications

base_dir = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(base_dir, 'Images')
csv_path = os.path.join(base_dir, 'Labels.csv')
results_dir = os.path.join(base_dir, 'results')
resized_folder = os.path.join(results_dir, 'images_resized')

for folder in [results_dir, resized_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

if not os.path.exists(csv_path):
    exit()

df = pd.read_csv(csv_path)

if 'Unnamed: 4' in df.columns:
    df.drop(columns=['Unnamed: 4'], inplace=True)

# Data Filtering: Selecting high-quality images and mapping diagnosis to binary labels
filtered_df = df[df["Quality Score"] >= 5].copy()
filtered_df["image_path"] = filtered_df["Image Name"].apply(lambda x: os.path.join(images_folder, x))
filtered_df["label_numeric"] = filtered_df["Label"].map({"GON+": 1, "GON-": 0})

# Image Preprocessing: Resizing fundus images to 224x224 for MobileNetV2 compatibility
for index, row in filtered_df.iterrows():
    save_path = os.path.join(resized_folder, row["Image Name"])
    if not os.path.exists(save_path):
        try:
            if os.path.exists(row["image_path"]):
                img = Image.open(row["image_path"]).convert("RGB").resize((224, 224))
                img.save(save_path)
        except Exception as e:
            print(f"Error: {e}")

filtered_df.to_csv(os.path.join(results_dir, 'glaucoma_clean_dataset.csv'), index=False)

# Patient-wise Splitting: Ensuring no data leakage by keeping images from the same patient in one set
patients = filtered_df["Patient"].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
train_df = filtered_df[filtered_df["Patient"].isin(train_patients)]
test_df = filtered_df[filtered_df["Patient"].isin(test_patients)]

train_df.to_csv(os.path.join(results_dir, "train_dataset.csv"), index=False)
test_df.to_csv(os.path.join(results_dir, "test_dataset.csv"), index=False)

def load_arrays(df_subset):
    X, y = [], []
    for _, row in df_subset.iterrows():
        img_path = os.path.join(resized_folder, row["Image Name"])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            X.append(np.array(img) / 255.0)
            y.append(row["label_numeric"])
    return np.array(X), np.array(y)

X_train, y_train = load_arrays(train_df)

# Transfer Learning: Initializing MobileNetV2 with ImageNet weights and adding custom classification layers
base_model = applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# Persistence: Saving the trained model weights for evaluation and dashboard deployment
model.save(os.path.join(results_dir, "glaucoma_model.h5"))