import os
import cv2
import json
import logging
from ultralytics import YOLO
from log_config import setup_logging

def create_incremental_folder(base_path, folder_prefix="video_results"):
    """
    Crée un dossier de la forme `video_resultsN` dans `base_path`
    en incrémentant N tant qu'un dossier du même nom existe déjà.
    Retourne le chemin complet du dossier créé.
    """
    n = 1
    while True:
        folder_name = f"{folder_prefix}{n}"
        full_path = os.path.join(base_path, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        n += 1

def test_video_on_model(video_path, model_path):
    setup_logging()
    logging.info("Début du traitement de la vidéo.")

    # --- Chargement du modèle ---
    try:
        model = YOLO(model_path)
        logging.info(f"Classes du modèle : {model.names}")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        return

    # --- Ouverture de la vidéo en lecture ---
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Impossible de charger la vidéo {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    except Exception as e:
        logging.error(f"Erreur lors de l'initialisation de la capture vidéo : {e}")
        return

    # --- Création du dossier de sortie ---
    base_output_path = "yolo_result"
    os.makedirs(base_output_path, exist_ok=True)

    # Incrémentation du dossier video_resultsN
    video_results_path = create_incremental_folder(base_output_path, "video_results")
    logging.info(f"Dossier de sortie créé : {video_results_path}")

    # Dossier pour stocker les JSON
    json_path = os.path.join(video_results_path, "json_labels")
    os.makedirs(json_path, exist_ok=True)
    logging.info(f"Dossier JSON créé : {json_path}")

    # Chemin et initialisation de la vidéo en écriture
    output_video_path = os.path.join(video_results_path, "video_results.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- Lecture et traitement frame par frame ---
    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prédiction sur la frame
            frame_results = model.predict(
                source=frame,
                conf=0.5,
                save=False,   # On n'enregistre pas d'images via YOLO, on va le faire manuellement
                stream=True   # On récupère le flux de prédictions
            )

            free_spaces = 0
            occupied_spaces = 0

            # Liste qui stockera des dictionnaires par objet détecté
            objects_detected = []

            for result in frame_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])

                    # Couleur selon la classe (1=libre, 0=occupé)
                    color = (0, 255, 0) if cls == 1 else (0, 0, 255)

                    # Dessiner la boîte sur la frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Mise à jour des compteurs
                    if cls == 1:
                        free_spaces += 1
                    else:
                        occupied_spaces += 1

                    # Stockage des infos dans la liste, avec des clés explicites
                    objects_detected.append({
                        "class_id": cls,
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2
                    })

            # Sauvegarde d'un fichier JSON pour cette frame
            json_data = {
                "frame_index": frame_index,
                "objects": objects_detected
            }

            json_filename = os.path.join(json_path, f"frame_{frame_index:04d}.json")
            with open(json_filename, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4)

            # Ajouter le compteur global de places libres
            cv2.putText(frame, f"{free_spaces} Places libres",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 128, 0), 2)

            # Écriture de la frame annotée dans la vidéo de sortie
            out.write(frame)

            frame_index += 1

        logging.info(f"Vidéo annotée sauvegardée : {output_video_path}")

    except Exception as e:
        logging.error(f"Erreur pendant le traitement : {e}")

    finally:
        # Libération des ressources
        cap.release()
        out.release()
        logging.info("Traitement terminé.")

if __name__ == "__main__":
    # Définir les chemins
    video_path = r"video_voirie\test4.mp4"  # Chemin de la vidéo
    model_path = "yolo8_nano_selfdataset_5E8/weights/best.pt"  # Chemin du modèle YOLO

    test_video_on_model(video_path, model_path)
