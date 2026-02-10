from abc import ABC, abstractmethod
import cv2 as cv
import threading
import numpy as np
import time

class BaseCamera(ABC):
    """
        Clasa de bază pentru toate camerele.
        Ofera mecanism de FPS si interfata standard

        params:
            use_gpu: Folosesti GPU pentru preprocesare (True/False)
     """
    def __init__(self, use_gpu=False, process: bool = False):
        self.process = process
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # GPU 
        self.use_gpu = use_gpu

        # FPS
        self.last_time = time.time()
        self.fps = 0

        self._init_gpu()

    def _init_gpu(self):
        """Initializeaza GPU"""
        if not self.use_gpu:
            return
        
        try:
            _ = cv.cuda.getCudaEnabledDeviceCount()
        except Exception:
            self.use_gpu = False
            return
        
        if cv.cuda.getCudaEnabledDeviceCount() == 0:
            self.use_gpu = False
            return
        
        self.gpu_input = cv.cuda.GpuMat()

    # ----------------------------------------- API ------------------------------------------   

    @abstractmethod
    def start(self):
        """Start achizitie"""
        pass

    @abstractmethod
    def read_raw(self):
        """Returneaza frame brut"""
        pass

    @abstractmethod
    def stop(self):
        """Opreste camera"""
        pass

    # ---------------------------------------- UTILITIES ----------------------------

    def compute_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.fps = 1.0/dt if dt > 0 else 0
        self.last_time = now

    def read(self):
        frame = self.read_raw()
        self.compute_fps()
        return frame
    
    def camera_get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
        
    def is_running(self):
        return self.running
    
    def compute_gray(self, frame):
        """Aplica procesare GPU/CPU daca e activa"""
        if frame is None:
            return None
        
        if self.use_gpu:
            self.gpu_input.upload(frame)
            gpu_gray = cv.cuda.cvtColor(self.gpu_input, cv.COLOR_BGR2GRAY)
            return gpu_gray.download()
        else:
            return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # ------------------------------ THREAD-SAFE -----------------------------------

    def _thread_loop(self):
        while self.running:
            
            frame = self.read()
            if frame is not None:
                with self.lock:
                    self.frame = frame

    
    


# =================================================================================
#               Cameră de test care generează imagini artificiale 
# =================================================================================


class DummyCamera(BaseCamera):
    """ Camera de test, genereaza imagini artificiale"""
    def __init__(self, use_gpu=False):
        super().__init__(use_gpu)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._thread_loop)
        self.thread.daemon = True
        self.thread.start()

    def read_raw(self):
        # Imagine (480 x 640)
        img = np.zeros((480, 640 ,3), dtype=np.uint8)
        cv.putText(img, f"FPS: {self.fps:.2f}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        time.sleep(0.01)
        return img


    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None




# ==================================================================================
#                                           TEST
# ==================================================================================

if __name__ == "__main__":
    cam = DummyCamera()
    cam.start()

    try:
        while True:
            if cam.frame is not None:
                cv.imshow("Test Camera", cam.frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally: 
        cam.stop()
        cv.destroyAllWindows()

   
