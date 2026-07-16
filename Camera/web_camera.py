from .base_camera import BaseCamera
import cv2 as cv
import threading
import time
from Errors import CameraError, safe_call, logger

class WebCam(BaseCamera):
    def __init__(self, cam_id=0, use_gpu=False, width=None, height=None):
        super().__init__(use_gpu)
        self.cam_id = cam_id
        self.cap = None
        self.width = width
        self.height = height

    def start(self):
        if self.running:
            logger.warning(f"[WebCam] Camera {self.cam_id} rulează deja.")
            return

        # Sfat: Pe Windows, uneori adăugarea cv.CAP_DSHOW rezolvă pornirea lentă
        self.cap = cv.VideoCapture(self.cam_id)
        
        if not self.cap.isOpened():
            logger.error(f"[WebCam] Nu poate fi deschisă camera ID {self.cam_id}")
            raise CameraError(f"Nu s-a putut inițializa WebCam cu ID-ul {self.cam_id}")
        
        # Setează dimensiunile dacă sunt specificate
        if self.width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
            
        # Verifică ce rezoluție a aprobat de fapt hardware-ul
        actual_w = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"[WebCam] Camera {self.cam_id} inițializată hardware la: {actual_w}x{actual_h}")

        self.running = True
        self.thread = threading.Thread(target=self._thread_loop, daemon=True) 
        self.thread.start()

    def read_raw(self):
        # Verificare rapidă thread-safe în caz că se dă stop din exterior
        if not self.running or self.cap is None:
            return None
            
        if not self.cap.isOpened():
            logger.warning("[WebCam] Încercare de citire de la o cameră închisă.") 
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("[WebCam] Cadru gol primit sau cameră deconectată.")
            return None
            
        return frame

    def stop(self):
        if not self.running:
            return

        logger.info(f"[WebCam] Se oprește camera {self.cam_id}...")
        self.running = False
        
        # Așteptăm ca thread-ul să iasă din bucla _thread_loop în mod natural
        if self.thread:
            self.thread.join(timeout=1.5)
            if self.thread.is_alive():
                logger.error(f"[WebCam] Thread-ul pentru camera {self.cam_id} a refuzat să moară la timp!")
            self.thread = None

        # Eliberăm hardware-ul DOAR după ce thread-ul s-a oprit complet
        if self.cap:
            self.cap.release()
            logger.info(f"[WebCam] Resurse hardware eliberate pentru camera {self.cam_id}.")
        
        self.cap = None
        
        with self.lock:
            self.frame = None  # Curățăm și ultimul cadru ca să nu trimitem mizerii la GUI
