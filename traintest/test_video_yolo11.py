import os
from ultralytics import YOLO

def test_video_on_model(video_path, model_path, output_path):
    """
    Teste une vidéo sur le modèle YOLOv8 entraîné.

    :param video_path: Chemin de la vidéo à tester.
    :param model_path: Chemin du fichier du modèle YOLOv8 entraîné (.pt).
    :param output_path: Chemin pour sauvegarder la vidéo avec les prédictions.
    """
    # Charger le modèle YOLOv8 entraîné
    model = YOLO(model_path)

    # Effectuer une prédiction sur la vidéo
    results = model.predict(
        source=video_path,  # Chemin de la vidéo
        save=True,          # Sauvegarder les résultats
        save_txt=True,      # Sauvegarder les prédictions en format texte
        project=output_path,  # Dossier de sortie
        name="video_results", # Nom du dossier de sortie
        conf=0.7        # Seuil de confiance
    )
    print(f"Vidéo testée. Résultats sauvegardés dans : {os.path.join(output_path, 'video_results')}")

if __name__ == "__main__":
    # Chemin vers la vidéo d'entrée
    video_path = "D:/Desktop/ERWAN/PFE/video.mp4"  # Remplacez par votre chemin réel

    # Chemin vers le modèle YOLOv8 entraîné
    model_path = "D:/Desktop/ERWAN/PFE/parking_model/yolo8_nano_selfdataset_5E8/weights/best.pt"  # Remplacez par le chemin réel

    # Chemin vers le répertoire pour sauvegarder les résultats
    output_path = "D:/Desktop/ERWAN/PFE/yolo11_result"

    # Tester la vidéo sur le modèle
    test_video_on_model(video_path, model_path, output_path)
