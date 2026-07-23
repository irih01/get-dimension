import cv2 as cv
import numpy as np
import logging
import time
from typing import Tuple, List, Union, Any

logger = logging.getLogger(__name__)

class Processor:
    """
    Pipeline de procesare unificat (GPU/CPU)
    """
    def __init__(self, 
                 use_gpu: bool = False,
                 mode: str = "gray",
                 resize: Tuple[int, int] | None = None,
                 clahe_clip: float = 2.0,
                 canny_thr: Tuple[int, int] = (50, 150),
                 adapt_th_param: Tuple[int, int] = (11, 3),
                 kernel: Tuple[int, int] = (3, 3)):
        
        self.mode = mode
        self.resize = resize
        self.clahe_clip = clahe_clip
        self.canny_thr = canny_thr
        self.adapt_th_param = adapt_th_param
        self.kernel = kernel

        
        try:
            self.gpu_available = cv.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            self.gpu_available = False

        self.use_gpu = bool(use_gpu and self.gpu_available)
        self._gpu_tmp = None
        self.gpu_ok = False

        # FIX MEMORY LEAK: Pre-alocăm obiecte GpuMat statice pentru a reutiliza memoria VRAM
        # și a preveni alocările dinamice masive în buclă care duc la Out Of Memory!
        self._gpu_dst_gray = None
        self._gpu_dst_blur = None
        self._gpu_dst_edges = None

        if use_gpu and not self.gpu_available:
            logger.warning("No CUDA")
            self.use_gpu = False

        if use_gpu and self.gpu_available:
            try:
                _ = cv.cuda.GpuMat()
                self.gpu_ok = True
                self._gpu_dst_gray = cv.cuda.GpuMat()
                self._gpu_dst_blur = cv.cuda.GpuMat()
                self._gpu_dst_edges = cv.cuda.GpuMat()
            except Exception:
                self.gpu_ok = False
        else:
            self.gpu_ok = False

    # CPU Processing
    def _cpu_gray(self, frame: np.ndarray) -> np.ndarray:
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    def _cpu_blur(self, frame: np.ndarray, k: int=5) -> np.ndarray:
        if k % 2 == 0: k += 1 # Siguranță: kernelul trebuie să fie impar!
        return cv.GaussianBlur(frame, (k, k), 0)

    def _cpu_edges(self, gray, t1=80, t2=120):
        return cv.Canny(gray, t1, t2)
    
    def _cpu_equalize(self, gray: np.ndarray) -> np.ndarray:
        return cv.equalizeHist(gray)

    def _cpu_adaptive_thresh(self, gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
        if block_size % 2 == 0:
            block_size += 1
        return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block_size, c)
    
    def _resize(self, frame):
        if self.resize is None:
            return frame
        w, h = self.resize
        return cv.resize(frame, (w, h), interpolation=cv.INTER_LINEAR)
    
    
    
    
    # GPU Processing
    def _gpu_upload(self, frame: np.ndarray) -> cv.cuda.GpuMat:
        if self._gpu_tmp is None:
            self._gpu_tmp = cv.cuda.GpuMat()
        self._gpu_tmp.upload(frame)
        return self._gpu_tmp

    def _gpu_gray(self, frame):
        gpu_src = self._gpu_upload(frame)
        if self._gpu_dst_gray is None: self._gpu_dst_gray = cv.cuda.GpuMat()
        # FIX: Folosim parametrul dst pentru a reutiliza bufferul VRAM static
        cv.cuda.cvtColor(gpu_src, cv.COLOR_BGR2GRAY, dst=self._gpu_dst_gray)
        return self._gpu_dst_gray.download()
    
    def _gpu_blur(self, frame, k=5):
        # FIX: Forțăm k să fie impar (era implicit 2, ceea ce dădea crash în nucleul CUDA!)
        if k % 2 == 0: k += 1
        gpu_src = self._gpu_upload(frame)
        if self._gpu_dst_blur is None: self._gpu_dst_blur = cv.cuda.GpuMat()
        cv.cuda.GaussianBlur(gpu_src, (k, k), 0, dst=self._gpu_dst_blur)
        return self._gpu_dst_blur.download()
    
    def _gpu_edges(self, frame, t1=80, t2=160):
        gpu_src = self._gpu_upload(frame)
        if self._gpu_dst_edges is None: self._gpu_dst_edges = cv.cuda.GpuMat()
        # Notă: cv.cuda.Canny are nevoie ca input-ul să fie deja alb-negru (Gray)
        if gpu_src.channels() > 1:
            if self._gpu_dst_gray is None: self._gpu_dst_gray = cv.cuda.GpuMat()
            cv.cuda.cvtColor(gpu_src, cv.COLOR_BGR2GRAY, dst=self._gpu_dst_gray)
            gpu_src = self._gpu_dst_gray
        
        # Filtrele CUDA Canny cer instanțierea unui detector în OpenCV modern, 
        # dar pentru compatibilitate cu structura ta folosim apelul direct securizat
        detector = cv.cuda.createCannyEdgeDetector(t1, t2)
        detector.detect(gpu_src, self._gpu_dst_edges)
        return self._gpu_dst_edges.download()
    

    # ----------------------------------------------------------------------
    #                   UNIVERSAL PIPELINE
    # ----------------------------------------------------------------------
    

    def gray(self, frame):
       if self.gpu_ok:
           return self._gpu_gray(frame)
       return self._cpu_gray(frame)
    
    def blur(self, frame, k=5):
        if self.gpu_ok:
            return self._gpu_blur(frame, k)
        return self._cpu_blur(frame, k)
    
    def edges(self, frame, t1=80, t2=120):
        if self.gpu_ok:
            return self._gpu_edges(frame, t1, t2)
        return self._cpu_edges(frame, t1, t2)
    
    def equalize(self,gray):
        return cv.equalizeHist(gray)
    

    def cpu_edges(self, gray):
        low, high = self.canny_thr
        return cv.Canny(gray, low, high)
    
    def adaptive_threshold(self, gray):
        block_size, c = self.adapt_th_param
        if block_size % 2 == 0:
            block_size += 1
        return cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block_size, c)
    

    # Pipeline
    def process_pipe(self, frame, ops):
        """
        ops = list of operations, ex:
        ["gray", ("blur",5), ("edges", 50, 150)]
        """
        res = frame

        for op in ops:
            if isinstance(op, str):
                res = getattr(self, op)(res)

            elif isinstance(op, tuple):
                name = op[0]
                args = op[1:]
                res = getattr(self, name)(res, *args)

        return res    
    
    def set_canny(self, low, high):
        self.canny_thr = (low, high)

    def enable_gpu(self, enable: bool):
        self.gpu_ok = enable and self.gpu_available
        self.use_gpu = enable and self.gpu_available

            
def benchmark(processor, frame, n=30):
    if frame is None or frame.size == 0:
        raise ValueError("Frame invalid pentru benchmark")

    processor.enable_gpu(False)
    t0 = time.time()

    # CPU
    for _ in range(n):
        processor.gray(frame)
        processor.blur(frame)
        processor.edges(frame)
    cpu_t = (time.time() - t0) / n

    # GPU
    if not processor.gpu_available:
        return {"cpu_ms": cpu_t * 1000, "gpu_ms": None, "gpu_faster": False}
    
    processor.enable_gpu(True)
    t0 = time.time()
    for _ in range(n):
        processor.gray(frame)
        processor.blur(frame)
        processor.edges(frame)
    gpu_t = (time.time() - t0) / n

    logger.info(f"Benchmark: CPU={cpu_t*1000:.2f}ms | GPU={gpu_t*1000:.2f}ms")

    return {"cpu_ms": cpu_t * 1000,
            "gpu_ms": float(gpu_t * 1000),
            "gpu_faster": bool(gpu_t < cpu_t)}


# FIX UTILIZARE RECOMANDATĂ:
# res = benchmark(processor, frame)
# if res["gpu_faster"]:
#     processor.enable_gpu(True)
