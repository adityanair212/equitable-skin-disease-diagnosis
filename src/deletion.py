import os
import glob

# Root directory
root_dir = r"C:\Users\adity\Desktop\Work\University\Project - Skin Disease Detection\Dataset\side_by_side_outputs"

# Supported image extensions
image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

# Loop through model folders
for model_folder in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_folder)
    if not os.path.isdir(model_path):
        continue

    print(f"\n🔍 Checking model: {model_folder}")

    # Loop through class subfolders
    for class_folder in os.listdir(model_path):
        class_path = os.path.join(model_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        print(f"  📂 Class: {class_folder}")

        # Check all files in class folder
        for file in os.listdir(class_path):
            if "_sidebyside" in file:
                sbs_path = os.path.join(class_path, file)
                base_name = file.replace("_sidebyside", "")
                
                # Try all supported image extensions
                found_original = False
                for ext in image_exts:
                    original_path = os.path.join(class_path, base_name.split('.')[0] + ext)
                    if os.path.exists(original_path):
                        found_original = True
                        print(f"    🗑️ Deleting: {sbs_path} (original found: {os.path.basename(original_path)})")
                        os.remove(sbs_path)
                        break

                if not found_original:
                    print(f"    ⚠️ No original found for: {file} — skipped")
