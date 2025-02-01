import logging
import os
from datetime import datetime

def setup_logging():
    """ Configure les logs en fonction de la date et de l'heure actuelle et s'assure que les dossiers sont bien créés. """

    # Obtenir la date et l'heure actuelles
    now = datetime.now()
    year_month = f"{now.year}_{now.month:02d}"  # Ex: "2025_01"
    day = f"{now.day:02d}"  # Ex: "30"
    log_dir = os.path.join("logs", year_month, day)  # Chemin absolu

    log_filename = f"log_{now.hour}h_{now.minute}min_{now.second}sec.log"
    log_path = os.path.join(log_dir, log_filename)

    try:
        # Création des répertoires
        os.makedirs(log_dir, exist_ok=True)
        print(f"Dossier de log créé ou déjà existant : {log_dir}")  # Debugging

    except Exception as e:
        print(f"Erreur lors de la création du dossier de log : {e}")

    # Configuration du logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logger initialisé avec succès. Logs enregistrés dans {log_path}")
