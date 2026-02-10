from .base_camera import BaseCamera
import cv2 as cv
import threading
from Errors import CameraError, safe_call, logger



class WebCam(BaseCamera):
    def __init__(self, cam_id = 0, use_gpu=False, width=None, height=None):
        super().__init__(use_gpu)

        self.cam_id = cam_id
        self.cap = None
        self.width = width
        self.height = height



    def start(self):
        self.cap = cv.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            logger.error(f"[WebCam] Nu poate fi deschis camera ID {self.cam_id}")
            return
        
        if self.width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(f"[WebCam] Camera {self.cam_id} setata {self.width} x {self.height}")

        self.running = True
        self.thread = threading.Thread(target = self._thread_loop, daemon=True) 
        self.thread.start()



    def read_raw(self):
        if not self.cap or not self.cap.isOpened():
            logger.warning("[WebCam] Incercat citirea de la o camera inchisa") 
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("[WebCam] Camera neconectata sau frame esuat")
            return None
        return frame
    
  

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        if self.cap:
            self.cap.release()
            logger.info(f"[WebCam] Camera {self.cam_id} oprita.")
        
        self.cap = None

