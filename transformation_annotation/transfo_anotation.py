import os
import json

# Chemins
coco_file = "C:/Users/Utilisateur/.cache/kagglehub/datasets/ammarnassanalhajali/pklot-dataset/versions/1/valid/_annotations.coco.json"
output_dir = "C:/Users/Utilisateur/.cache/kagglehub/datasets/ammarnassanalhajali/pklot-dataset/versions/1/valid/labels"

# Créer le dossier de sortie
os.makedirs(output_dir, exist_ok=True)

# Charger les annotations COCO
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Extraire les informations des annotations COCO
for image in coco_data['images']:
    image_id = image['id']
    file_name = os.path.splitext(image['file_name'])[0]
    label_file_path = os.path.join(output_dir, f"{file_name}.txt")

    # Trouver toutes les annotations pour cette image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # Écrire les annotations au format YOLO
    with open(label_file_path, 'w') as label_file:
        for ann in annotations:
            category_id = ann['category_id'] - 1  # YOLO utilise des index 0-based
            bbox = ann['bbox']
            x_center = (bbox[0] + bbox[2] / 2) / image['width']
            y_center = (bbox[1] + bbox[3] / 2) / image['height']
            width = bbox[2] / image['width']
            height = bbox[3] / image['height']
            label_file.write(f"{category_id} {x_center} {y_center} {width} {height}/n")