import os

# Répertoire contenant vos fichiers de labels
labels_dir = "C:/Users/Utilisateur/.cache/kagglehub/datasets/ammarnassanalhajali/pklot-dataset/versions/1/test/labels"

# Parcourir tous les fichiers de labels
for file_name in os.listdir(labels_dir):
    file_path = os.path.join(labels_dir, file_name)

    # Ouvrir et corriger le contenu
    with open(file_path, "r") as file:
        lines = file.read().replace("/n", "\n").strip()  # Supprimer les "/n" et nettoyer

    # Réécrire le fichier corrigé
    with open(file_path, "w") as file:
        file.write(lines)

print("Tous les fichiers de labels ont été corrigés.")
