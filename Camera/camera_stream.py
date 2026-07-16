import threading
import time
from Errors import CameraError, logger, safe_call


class CameraStream:
    """
    Thread care preia ultimul frame si il face disponibil GUI-ului.
    """
    def __init__(self, cam, poll_interval=0.01 , fps_limit = 60):

        self.camera = cam
        self.poll_interval = poll_interval
        self.frame = None

        self.running = False

        self.thread = None
        self.lock = threading.Lock()

        self.fps_limit = fps_limit
        self.last_timne = time.time()
        self.fps = 0.0



    def _loop(self):
            """
            Ruleaza intr-un thread separat pentru GUI
            """

            #delay = 1.0 / self.fps_limit if self.fps_limit > 0 else 0

            while self.running:
                # FIX: În loc să forțăm camera să facă o nouă citire hardware (ceea ce provoacă lag),
                # extragem direct ultimul cadru capturat deja în background de thread-ul camerei.
                if hasattr(self.camera, "camera_get_frame"):
                    frame = self.camera.camera_get_frame()
                elif hasattr(self.camera, "read"):
                    frame = self.camera.read()
                else:
                    frame = self.camera.read_raw()

                if frame is not None:
                    with self.lock:
                        self.frame = frame

                    now = time.time()
                    dt = now - self.last_timne
                    if dt > 0:
                        self.fps = 1.0 / dt
                    self.last_timne = now

                # Mici pauze ca să nu sufoce procesorul, adaptat la intervalul cerut
                time.sleep(self.poll_interval)




    def start(self):
        if hasattr(self.camera, "start"):
            try:
                self.camera.start()
            except Exception as e:
                raise CameraError(f"Nu poate porni stream-ul {e}")

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Stream pornit")


    @safe_call(default_return=None)
    def stream_get_frame(self):
        """ Returneaza (ret, frame)"""
        with self.lock:
            if self.frame is None:
                logger.warning("Nu pot citi frame de la camera")
                return False, None
            return True, self.frame.copy()


    def stop(self):
        """Opreste thread-ul camerei"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None # Curățăm referința thread-ului mort
            
        # FIX: Eliminăm apelul redundant self.camera.stop() de aici, deoarece
        # de închiderea hardware-ului se ocupă deja în mod centralizat CameraManager.
        logger.info("Stream Oprit")


    def release(self):
        """Alias pentru stop"""
        self.stop()
