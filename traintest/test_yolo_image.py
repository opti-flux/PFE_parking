import os
from ultralytics import YOLO

def test_images_on_model(image_folder, model_path, output_path):
    """
    Teste un dossier d'images sur le modèle YOLOv8 entraîné.

    :param image_folder: Chemin vers le dossier contenant les images à tester.
    :param model_path: Chemin du fichier du modèle YOLOv8 entraîné (.pt).
    :param output_path: Chemin pour sauvegarder les images avec les prédictions.
    """
    # Charger le modèle YOLOv8 entraîné
    model = YOLO(model_path)

    # Effectuer une prédiction sur les images
    results = model.predict(
        source=image_folder,  # Chemin vers le dossier d'images
        save=True,            # Sauvegarder les résultats
        save_txt=True,        # Sauvegarder les prédictions en format texte
        project=output_path,  # Dossier de sortie
        name="image_results", # Nom du dossier de sortie
        conf=0.5              # Seuil de confiance
    )
    print(f"Images testées. Résultats sauvegardés dans : {os.path.join(output_path, 'image_results')}")

if __name__ == "__main__":
    # Chemin vers le dossier contenant les images
    image_folder = "D:/Desktop/ERWAN/PFE/images"  # Remplacez par votre chemin réel

    # Chemin vers le modèle YOLOv8 entraîné
    model_path = "D:/Desktop/ERWAN/PFE/parking_model/yolo8_nano_100_E_first_dataset3/weights/best.pt"  # Remplacez par le chemin réel

    # Chemin vers le répertoire pour sauvegarder les résultats
    output_path = "D:/Desktop/ERWAN/PFE/yolo_images_result"

    # Tester les images sur le modèle
    test_images_on_model(image_folder, model_path, output_path)
