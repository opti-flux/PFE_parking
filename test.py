from ultralytics import YOLO
import cv2
import os

# Charger le modèle YOLO
model = YOLO('yolov8s.pt')  # Utilisez le modèle YOLOv8 pré-entraîné

# Effectuer des prédictions
results = model('frame_0001.jpg', conf=0.03,imgsz = 1280)  # Réduisez le seuil de confiance si nécessaire

# Créer un dossier pour enregistrer les images annotées
output_folder = 'cars_only'
os.makedirs(output_folder, exist_ok=True)

# Filtrer et annoter uniquement les voitures
for result in results:
    # Charger l'image d'origine
    img = result.orig_img  # Image d'entrée non modifiée
    
    # Filtrer pour ne conserver que les boîtes avec la classe "car" (classe 2 pour COCO)
    car_boxes = [box for box in result.boxes if int(box.cls[0]) == 2]
    print(f"Voitures détectées : {len(car_boxes)}")
    
    # Annoter l'image avec uniquement les voitures détectées
    for box in car_boxes:
        coords = box.xyxy[0]  # Coordonnées des boîtes (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, coords)
        confidence = box.conf[0]  # Confiance de la détection
        label = "car"  # Nom de la classe (ou utilisez `model.names[2]`)
        
        # Dessiner les boîtes sur l'image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Boîte verte
        cv2.putText(img, f"{label} {confidence:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Texte avec confiance

    # Enregistrer l'image annotée
    output_path = os.path.join(output_folder, 'frame_0001_cars_only.jpg')
    cv2.imwrite(output_path, img)  # Enregistrer l'image avec les annotations
    print(f"Image annotée enregistrée dans : {output_path}")
