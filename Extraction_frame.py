import cv2
import os

# Chemin du dossier contenant les vidéos
videos_folder = "video_voirie"

# Répertoire de sortie principal pour les frames
output_root_folder = "extracted_frames"
os.makedirs(output_root_folder, exist_ok=True)

# Nombre de frames à extraire par seconde
frames_per_second = 5  # Modifier cette valeur si nécessaire

# Lister tous les fichiers dans le dossier
video_files = [f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

if not video_files:
    print("Aucune vidéo trouvée dans le dossier.")
    exit()

# Parcourir chaque vidéo
for video_file in video_files:
    video_path = os.path.join(videos_folder, video_file)
    video_name = os.path.splitext(video_file)[0]  # Nom du fichier sans extension

    # Créer un dossier pour les frames de cette vidéo
    output_folder = os.path.join(output_root_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Charger la vidéo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erreur : Impossible de lire la vidéo {video_file}.")
        continue

    # Récupérer les FPS réels de la vidéo
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculer l'intervalle entre les frames à extraire
    frame_interval = max(1, video_fps // frames_per_second)

    print(f"Traitement de la vidéo : {video_file}")
    print(f"FPS vidéo : {video_fps}, extraction de {frames_per_second} FPS (intervalle : {frame_interval})")

    frame_count = 0
    extracted_count = 0

    # Lire les frames de la vidéo
    while True:
        ret, frame = cap.read()
        if not ret:  # Fin de la vidéo
            break

        # Extraire la frame si elle correspond à l'intervalle
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Enregistré : {frame_filename}")
            extracted_count += 1

        frame_count += 1

    # Libérer les ressources
    cap.release()
    print(f"Extraction terminée pour {video_file}. {extracted_count} frames extraites.")

print("Traitement terminé pour toutes les vidéos.")
