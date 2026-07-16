from .camera_stream import CameraStream
from .flir_camera import FLIRCamera
from .web_camera import WebCam
from Errors import CameraError, safe_call, logger

class CameraManager:
    """
    Manager unificat pentru camere.
    
    Principii:
    ----------
    • create_camera() -> DOAR instanțiază camera (nu pornește nimic)
    • create_stream() -> crează stream, nu îl pornește
    • start()         -> pornește stream-ul (și camera)
    • stop()          -> oprește TOT (stream + camera)
    • get_frame()     -> returnează ultimul frame
 
    """
    def __init__(self):
        self.stream = None
        self.camera = None
        self.processor = None # Optonal: GPU/CPU

    def create_camera(self, cam_type="webcam", **kwargs):
        """
        params:
        cam_type: "webcam" | "flir" \n
        webcam: cam_id , use_gpu, width=, height \n
        flir: cam_idx, px_format
        """
        cam_type = cam_type.lower().strip()

        if self.stream or self.camera:
            logger.warning("[CameraManager] Se încearcă recreerea camerei. Se forțează oprirea fluxului vechi.")
            self.stop()

        if cam_type == "webcam":
             self.camera = WebCam(**kwargs)
        elif cam_type == "flir":
            self.camera = FLIRCamera(**kwargs)
        else:
            raise CameraError(f"Tip camera necunoscuta: {cam_type}")
        return self.camera

    def create_stream(self, camera, processor=None, fps=60):
        if not self.camera:
            raise CameraError("Nu exista camera creata. Apeleaza create")

        self.processor = processor
        self.stream = CameraStream(camera, poll_interval=1.0/fps)

        if self.processor:
            self.stream.processor = processor

        logger.info("CameraMnager Stream creat.")
        return self.stream

    def start(self):
        if not self.stream:
            raise RuntimeError("Streamul nu a fost creat")
        self.stream.start()
        logger.info("CamManager pornit")

    def stop(self):
        error_messages = []

        if self.stream:
            try:
                self.stream.stop()
            except Exception as e:
                error_messages.append(f"Eroare la oprire stream: {e}")
            finally:
                self.stream = None

        if self.camera:
            try:
                self.camera.stop()
            except Exception as e:
                error_messages.append(f"Eroare la oprire camera: {e}")
            finally:
                self.camera = None

        if error_messages:
            combined_error = " | ".join(error_messages)
            logger.error(f"[CameraManager] Probleme la închiderea resurselor: {combined_error}")
            raise CameraError(f"[CameraManager] Defecțiune la curățarea resurselor: {combined_error}")

    def get_frame(self):
        return None if not self.stream else self.stream.stream_get_frame()

    def is_running(self):
        return self.stream is not None
