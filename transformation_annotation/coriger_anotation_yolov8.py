import os

def clean_annotations(labels_dir):
    """
    Vérifie et nettoie les fichiers d'annotations dans un dossier donné.
    Supprime les segments et garde uniquement les bounding boxes au format YOLO.
    
    Args:
        labels_dir (str): Chemin vers le dossier contenant les fichiers d'annotations (.txt).
    """
    for filename in os.listdir(labels_dir):
        file_path = os.path.join(labels_dir, filename)
        if not filename.endswith(".txt"):
            continue  # Ignorer les fichiers non texte

        cleaned_lines = []
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # Format YOLO attendu : <class_id> <x_center> <y_center> <width> <height>
                    try:
                        # Vérifier que tous les éléments sont des nombres
                        class_id, x_center, y_center, width, height = map(float, parts)
                        cleaned_lines.append(line)
                    except ValueError:
                        print(f"Ligne invalide ignorée dans {file_path}: {line.strip()}")
                else:
                    print(f"Ligne ignorée (probablement un segment) dans {file_path}: {line.strip()}")

        # Réécrire le fichier avec les lignes nettoyées
        with open(file_path, "w") as file:
            file.writelines(cleaned_lines)
        print(f"Fichier nettoyé : {file_path}")

# Chemins vers les dossiers contenant les fichiers d'annotations
train_labels_dir = "D:/Desktop/ERWAN/PFE/hady_dataset/train/labels"
valid_labels_dir = "D:/Desktop/ERWAN/PFE/hady_dataset/valid/labels"

# Nettoyer les annotations pour les ensembles d'entraînement et de validation
print("Nettoyage des annotations pour l'ensemble d'entraînement...")
clean_annotations(train_labels_dir)

print("Nettoyage des annotations pour l'ensemble de validation...")
clean_annotations(valid_labels_dir)

print("Nettoyage terminé !")
