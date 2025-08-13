import os
import cv2
import numpy as np
from tqdm import tqdm

gradcam_root = 'gradcam_outputs'
original_root = 'processed_dataset/test'
output_root = 'side_by_side_outputs'
font = cv2.FONT_HERSHEY_SIMPLEX

# Add text label above image
def add_label(image, label_text):
    label_height = 30
    labeled_img = np.full((image.shape[0] + label_height, image.shape[1], 3), 255, dtype=np.uint8)
    labeled_img[label_height:] = image
    cv2.putText(labeled_img, label_text, (10, 22), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return labeled_img

# Attempt to match partial filename (ignore disease suffix)
def find_original_image(original_dir, gradcam_filename):
    target_base = gradcam_filename.split('_')[0]  # e.g., "492" or "padufes"
    for fname in os.listdir(original_dir):
        if fname.startswith(target_base) and fname.endswith(('.png', '.jpg', '.jpeg')):
            return os.path.join(original_dir, fname)
    return None

# Generate side-by-side comparisons
for model_name in os.listdir(gradcam_root):
    for class_name in os.listdir(os.path.join(gradcam_root, model_name)):
        gradcam_class_dir = os.path.join(gradcam_root, model_name, class_name)
        original_class_dir = os.path.join(original_root, class_name)
        save_class_dir = os.path.join(output_root, model_name, class_name)
        os.makedirs(save_class_dir, exist_ok=True)

        for fname in tqdm(os.listdir(gradcam_class_dir), desc=f"{model_name}/{class_name}"):
            gradcam_path = os.path.join(gradcam_class_dir, fname)
            original_path = find_original_image(original_class_dir, fname)
            save_path = os.path.join(save_class_dir, fname.replace(".png", "_sidebyside.png"))

            if not original_path or not os.path.exists(original_path):
                print(f"❌ Missing original for: {fname}")
                continue

            # Read and resize
            original = cv2.resize(cv2.imread(original_path), (224, 224))
            gradcam = cv2.resize(cv2.imread(gradcam_path), (224, 224))

            labeled_orig = add_label(original, "Original")
            labeled_cam = add_label(gradcam, "Grad-CAM")
            combined = np.hstack((labeled_orig, labeled_cam))

            cv2.imwrite(save_path, combined)
