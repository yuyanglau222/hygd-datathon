import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')
model_path = os.path.join(results_dir, "glaucoma_model.h5")
test_csv = os.path.join(results_dir, "test_dataset.csv")
resized_folder = os.path.join(results_dir, 'images_resized')

if not os.path.exists(model_path):
    print(f"❌ Error: Could not find {model_path}. Please run glaucoma.py first.")
    exit()

model = tf.keras.models.load_model(model_path)
test_df = pd.read_csv(test_csv)

def load_test_arrays(df):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(resized_folder, row["Image Name"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            X.append(np.array(img) / 255.0)
            y.append(row["label_numeric"])
    return np.array(X), np.array(y)

X_test, y_test = load_test_arrays(test_df)

# Prediction: Generate probability scores and binary classes for the test set
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n" + "="*40)
print("--- Classification Report ---")
target_names = ['Normal (GON-)', 'Glaucoma (GON+)']
print(classification_report(y_test, y_pred, target_names=target_names))
print("="*40 + "\n")

# Performance Visualization: Plotting ROC Curve and Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate (1 - Specificity)')
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_title('Receiver Operating Characteristic (ROC)')
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=target_names, yticklabels=target_names)
ax2.set_title('Confusion Matrix - Glaucoma Detection')
ax2.set_ylabel('True Clinical Diagnosis')
ax2.set_xlabel('Model Prediction')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "model_performance_summary.png"))
plt.show()

# Grad-CAM Algorithm: Computes heatmap to visualize areas influencing the AI's decision
def make_gradcam_heatmap(img_array, full_model, last_conv_layer_name='out_relu'):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    base_model = full_model.layers[0]
    conv_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = conv_model(img_tensor)
        tape.watch(conv_outputs)
        x = base_outputs
        for layer in full_model.layers[1:]:
            x = layer(x)
        preds = x
    grads = tape.gradient(preds, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Interpretability: Superimposing the Grad-CAM heatmap onto the original image
sample_idx = 0
sample_img = X_test[sample_idx:sample_idx+1]
heatmap = make_gradcam_heatmap(sample_img, model)

original_img = (sample_img[0] * 255).astype("uint8")
heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_rgb, 0.4, 0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Scan")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("AI Focus Heatmap")
plt.imshow(heatmap, cmap='jet')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Decision Overlay")
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()