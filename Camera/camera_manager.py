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

        if self.camera:
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
        if self.stream:
            try:
                self.stream.stop()
            except Exception as e:
                raise CameraError(f"[CameraManager] Eroare la oprire stream: {e}")
            self.stream = None

        if self.camera:
            try:
                self.camera.stop()
            except Exception as e:
                raise CameraError(f"[CameraManager] Eroare la oprire camera: {e}")
            self.camera = None


    def get_frame(self):
        return None if not self.stream else self.stream.stream_get_frame()
    
    def is_running(self):
        return self.stream is not None

