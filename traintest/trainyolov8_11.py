import os

def create_dataset_yaml(dataset_path):
    """
    Crée un fichier de configuration YAML pour YOLOv8.
    """
    dataset_yaml = f"""
    path: {dataset_path}
    train: train/images
    val: valid/images

    # Noms des classes dans votre dataset
    names:
      0: occupied
      1: empty
    """
    yaml_path = os.path.join(dataset_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(dataset_yaml)
    print(f"Fichier YAML créé : {yaml_path}")
    return yaml_path

def install_ultralytics():
    """
    Installe la bibliothèque Ultralytics si elle n'est pas déjà installée.
    """
    try:
        import ultralytics
        print("Ultralytics est déjà installé.")
    except ImportError:
        print("Installation d'Ultralytics...")
        os.system("pip install ultralytics")

def train_yolo_model(yaml_path):
    """
    Entraîne un modèle YOLOv8 à partir d'un fichier YAML et des paramètres définis.
    """
    from ultralytics import YOLO

    # Charger un modèle YOLOv8 pré-entraîné
    model = YOLO("yolov8n.pt")  # Choisissez "yolov8n.pt", "yolov8s.pt", etc., selon vos besoins

    # Lancer l'entraînement
    model.train(
        data=yaml_path,  # Chemin vers le fichier YAML
        imgsz=1082,  # Taille des images
        epochs=50  ,  # Nombre d'époques
        batch=16,  # Taille du batch
        project="parking_model",  # Dossier de sauvegarde des résultats
        name="yolo8_nano_selfdataset_5E",  # Nom de l'expérience
        device=0,  # Spécifiez le GPU (ou "cpu" si pas de GPU disponible)
    )
    print("Entraînement terminé.")

if __name__ == "__main__":
    # Définir le chemin de votre dataset
    dataset_path = "D:/Desktop/ERWAN/PFE/test4_Erwan/reorganized_dataset"  # Remplacez par votre chemin réel

    # Étape 1 : Créer le fichier YAML
    yaml_path = create_dataset_yaml(dataset_path)

    # Étape 2 : Installer Ultralytics
    install_ultralytics()

    # Étape 3 : Entraîner le modèle YOLOv8
    train_yolo_model(yaml_path)
