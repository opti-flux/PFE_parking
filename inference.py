import os
import cv2
import json
import logging
from ultralytics import YOLO
from log_config import setup_logging


# Fonction pour créer un dossier unique en incrémentant un suffixe numérique si nécessaire
def create_incremental_folder(base_path, folder_prefix="video_results"):
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

    # Chargement du modèle YOLO avec les poids spécifiés
    try:
        model = YOLO(model_path)
        logging.info(f"Classes du modèle : {model.names}")
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        return

    # Ouverture de la vidéo pour traitement
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

    # Création des dossiers de sortie
    base_output_path = "yolo_result"
    os.makedirs(base_output_path, exist_ok=True)
    video_results_path = create_incremental_folder(base_output_path, "video_results")
    logging.info(f"Dossier de sortie créé : {video_results_path}")

    json_path = os.path.join(video_results_path, "json_labels")
    os.makedirs(json_path, exist_ok=True)
    logging.info(f"Dossier JSON créé : {json_path}")

    # Initialisation de la vidéo de sortie
    output_video_path = os.path.join(video_results_path, "video_results.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prédiction sur la frame
            frame_results = model.predict(
                source=frame,
                conf=0.85,
                save=False,
                stream=True
            )

            free_spaces = 0
            occupied_spaces = 0
            objects_detected = []

            for result in frame_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = "Libre" if cls == 1 else "Occupee"
                    color = (0, 255, 0) if cls == 1 else (0, 0, 255)

                    # Dessiner la boîte de détection sur la frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Mise à jour des compteurs selon la classe détectée
                    if cls == 1:
                        free_spaces += 1
                    else:
                        occupied_spaces += 1

                    # Stockage des informations détectées
                    objects_detected.append({
                        "class_id": cls,
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "confidence": confidence
                    })

                    # Ajouter un label au-dessus de la boîte englobante
                    label_text = f"{label} ({confidence:.2f})"
                    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_w, text_h = text_size
                    cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w + 5, y1), color, -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Sauvegarde des annotations dans un fichier JSON
            json_data = {
                "frame_index": frame_index,
                "objects": objects_detected
            }

            json_filename = os.path.join(json_path, f"frame_{frame_index:04d}.json")
            with open(json_filename, "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, indent=4)

            # Affichage du compteur global de places libres
            text = f"{free_spaces} Place libre" if free_spaces == 1 or free_spaces == 0 else f"{free_spaces} Places libres"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)

            # Écriture de la frame annotée dans la vidéo de sortie
            out.write(frame)
            frame_index += 1

        logging.info(f"Vidéo annotée sauvegardée : {output_video_path}")

    except Exception as e:
        logging.error(f"Erreur pendant le traitement : {e}")

    finally:
        cap.release()
        out.release()
        logging.info("Traitement terminé.")


if __name__ == "__main__":
    # Définition des chemins d'entrée
    video_path = r"video.mp4"
    model_path = "yolo11nano/weights/best.pt"

    # Lancement du traitement de la vidéo
    test_video_on_model(video_path, model_path)
