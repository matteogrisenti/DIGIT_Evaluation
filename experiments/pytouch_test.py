import cv2
from pytouch.sensors import DigitSensor
from pytouch.tasks import TouchDetect

def main():
    # 1. Definisci il percorso della tua immagine di test
    image_path = "saved_images/digit_2026-05-14_11-51-11.jpg"  # Cambia questo con il nome reale della tua immagine
    
    # Carica l'immagine usando OpenCV
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Errore: Impossibile caricare l'immagine '{image_path}'. Controlla il percorso.")
        return

    print("Immagine caricata correttamente. Tentativo di inizializzazione di PyTouch...")

    try:
        # 2. Inizializza il task di PyTouch
        # (Questo è il momento esatto in cui PyTouch proverà a connettersi ai server AWS di Meta)
        touch_detector = TouchDetect(DigitSensor, zoo_model="touchdetect_resnet18")
        
        print("Modello PyTouch caricato con successo!")
        
        # 3. Esegui l'inferenza sul frame
        print("Analisi dell'immagine in corso...")
        risultato = touch_detector(frame)
        
        # Gestione sicura dell'output (a seconda della versione esatta di PyTouch)
        if isinstance(risultato, tuple):
            is_touching, certainty = risultato
            print(f"\n--- RISULTATO ---")
            print(f"Contatto Rilevato: {is_touching}")
            print(f"Livello di Confidenza: {certainty:.2f}")
        else:
            print(f"\n--- RISULTATO ---")
            print(f"Contatto Rilevato: {risultato}")

    except Exception as e:
        # 4. Cattura e stampa l'eventuale errore del server
        print(f"\n[ERRORE PYTOUCH]: {e}")
        print("\nSe vedi un errore 'NoSuchBucket', significa che la libreria sta cercando di scaricare i pesi da un server che Meta ha chiuso.")

if __name__ == "__main__":
    main()