import cv2 as cv
import time
import os
import logging

# Importăm modulele din structura ta curată de directoare
from Camera.camera_manager import CameraManager
from Process.object_detector import ObjDetector
from Process.processor import Processor
from Errors.logger import logger

def main():
    logger.info("=== Pornire Sistem de Măsurare SIR (WebCam CPU Edition) ===")

    # 1. Configurare căi pentru matricea de calibrare
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calib_file = os.path.join(base_dir, "Calibration", "MultiMatrix.npz")
    
    if not os.path.exists(calib_file):
        logger.warning(f"[AVERTISMENT] Fișierul de calibrare {calib_file} lipsește!")
        logger.warning("Sistemul va rula DOAR în mod Pixeli (fără conversie în milimetri).")
        calib_file = None

    # 2. Inițializare Pipeline de Procesare (FORȚAT PE CPU)
    # Punem use_gpu=False ca să ignore complet nucleele CUDA și să ruleze stabil
    processor = Processor(use_gpu=False, canny_thr=(50, 150))
    
    detector = ObjDetector(
        use_aruco=True,       # Căutăm markerul ArUco pentru raportul mm/px dinamic
        marker_size=5.0,      # Dimensiunea reală a markerului tău în centimetri (ex: 5.0 cm)
        draw=True,            # Vrem să deseneze automat bbox și liniile de cotă pe ecran
        processor=processor
    )

    # 3. Inițializare și Configurare Hardware Cameră via CameraManager
    cam_manager = CameraManager()
    
    try:
        # Configuram camera comercială pe ID 0 tot pe modul CPU (use_gpu=False)
        cam_manager.create_camera(
            cam_type="webcam", 
            cam_id=0, 
            width=1280, 
            height=720, 
            use_gpu=False
        )
        
        # Creăm stream-ul dedicat de GUI limitat la 60 FPS ca să nu sufoce procesorul
        cam_manager.create_stream(fps=60)
        
        # Pornim motoarele! Hardware-ul și thread-ul secundar încep achiziția pe CPU
        cam_manager.start()
        
    except Exception as e:
        logger.error(f"Eroare fatală la inițializarea hardware-ului: {e}")
        cam_manager.stop()
        return

    # Fereastra grafică OpenCV pentru afișare în timp real
    win_name = "Sistem SIR - Masuratori live cu ArUco (CPU)"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    logger.info("Aplicația rulează pe CPU. Apasă tasta 'Q' în fereastra video pentru a închide programul.")

    # 4. Bucla Principală de Procesare Live
    try:
        while True:
            # Extragem ultimul cadru capturat în mod thread-safe din stream-ul camerei
            ret, frame = cam_manager.get_frame()
            
            if not ret or frame is None:
                # O mică pauză de siguranță ca să nu forțăm nucleul CPU dacă bufferul e momentan gol
                time.sleep(0.001)
                continue

            # Rulăm pipeline-ul de CPU: detecție marker, extragere contur, măsurare și desenare
            results = detector.process(frame)

            # Adăugăm un overlay rapid pentru FPS-ul general al aplicației
            if cam_manager.stream:
                current_fps = cam_manager.stream.fps
                cv.putText(frame, f"Stream FPS: {current_fps:.1f}", (20, frame.shape[0] - 20),
                           cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1, cv.LINE_AA)

            # Afișăm rezultatul final pompos pe ecran
            cv.imshow(win_name, frame)

            # Prindem evenimentele de tastatură de la utilizator
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Utilizatorul a cerut închiderea aplicației.")
                break
                
    finally:
        # 5. Eliberare Centralizată și Sigură a Resurselor
        logger.info("Se curăță memoria și se închid resursele hardware...")
        cam_manager.stop()
        cv.destroyAllWindows()
        logger.info("=== Sistem oprit cu succes. Noapte bună! ===")

if __name__ == "__main__":
    main()
