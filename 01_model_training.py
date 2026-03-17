import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications

# --- 1. DIRECTORY SETUP ---
# Detects the root folder: "D:/Downloads/hygd datathon"
base_dir = os.path.dirname(os.path.abspath(__file__))

# INPUTS: Looking for raw files in the root folder
images_folder = os.path.join(base_dir, 'Images')
csv_path = os.path.join(base_dir, 'Labels.csv')

# OUTPUTS: Creating a 'results' folder for all generated files
results_dir = os.path.join(base_dir, 'results')
resized_folder = os.path.join(results_dir, 'images_resized')

# Create necessary folders
for folder in [results_dir, resized_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

# --- 2. LOADING & CLEANING DATA ---
if not os.path.exists(csv_path):
    print(f"❌ Error: Cannot find {csv_path} in {base_dir}")
    exit()

df = pd.read_csv(csv_path)

# Drop unnecessary columns
if 'Unnamed: 4' in df.columns:
    df.drop(columns=['Unnamed: 4'], inplace=True)

# Quality filtering (>= 5)
filtered_df = df[df["Quality Score"] >= 5].copy()
filtered_df["image_path"] = filtered_df["Image Name"].apply(lambda x: os.path.join(images_folder, x))
filtered_df["label_numeric"] = filtered_df["Label"].map({"GON+": 1, "GON-": 0})

# --- 3. IMAGE PREPROCESSING (Resizing to 224x224) ---
print("Resizing images... saving to results/images_resized/")
for index, row in filtered_df.iterrows():
    save_path = os.path.join(resized_folder, row["Image Name"])
    if not os.path.exists(save_path):
        try:
            if os.path.exists(row["image_path"]):
                img = Image.open(row["image_path"]).convert("RGB").resize((224, 224))
                img.save(save_path)
            else:
                print(f"Skipping: {row['Image Name']} (Check if image is in {images_folder})")
        except Exception as e:
            print(f"Error processing {row['Image Name']}: {e}")

# Save the Cleaned Metadata to the results folder
filtered_df.to_csv(os.path.join(results_dir, 'glaucoma_clean_dataset.csv'), index=False)

# --- 4. DATA SPLITTING (Patient-wise) ---
patients = filtered_df["Patient"].unique()
train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)

train_df = filtered_df[filtered_df["Patient"].isin(train_patients)]
test_df = filtered_df[filtered_df["Patient"].isin(test_patients)]

# Save Train/Test CSVs to the results folder
train_df.to_csv(os.path.join(results_dir, "train_dataset.csv"), index=False)
test_df.to_csv(os.path.join(results_dir, "test_dataset.csv"), index=False)
print("✅ Datasets saved to results folder.")

# --- 5. MODEL TRAINING & SAVING ---
def load_arrays(df_subset):
    X, y = [], []
    for _, row in df_subset.iterrows():
        img_path = os.path.join(resized_folder, row["Image Name"])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            X.append(np.array(img) / 255.0)
            y.append(row["label_numeric"])
    return np.array(X), np.array(y)

print("Loading image arrays for training...")
X_train, y_train = load_arrays(train_df)

# Build MobileNetV2 Model
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

# Train the model
print("Starting training (15 Epochs)...")
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

# Save the final model file to the results folder
model.save(os.path.join(results_dir, "glaucoma_model.h5"))
print(f"🏁 DONE! Model and artifacts saved in: {results_dir}")