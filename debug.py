# debug_mini.py
import os

split_file = "splits/phc_train.txt"

# Lire toutes les lignes du fichier
with open(split_file, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total lines in {split_file}: {len(lines)}\n")

# Afficher les 5 premiers chemins et vérifier leur existence
for i, line in enumerate(lines[:5], 1):
    # Supposons que le format est "image_path,label_path"
    parts = line.split(",")
    img_path = parts[0]
    label_path = parts[1] if len(parts) > 1 else None

    img_exists = os.path.exists(img_path)
    label_exists = os.path.exists(label_path) if label_path else False

    print(f"{i}. Image: {img_path} (exists: {img_exists})")
    print(f"   Label: {label_path} (exists: {label_exists})\n")
