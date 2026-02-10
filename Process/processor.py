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

        if use_gpu and not self.gpu_available:
            logger.warning("No CUDA")
            self.use_gpu = False

        if use_gpu and self.gpu_available:
            try:
                _ = cv.cuda.GpuMat()
                self.gpu_ok = True
            except Exception:
                self.gpu_ok = False
        else:
            self.gpu_ok = False

    # CPU Processing
    def _cpu_gray(self, frame: np.ndarray) -> np.ndarray:
        return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    def _cpu_blur(self, frame: np.ndarray, k: int=5) -> np.ndarray:
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
        gpu = self._gpu_upload(frame)
        gpu = cv.cuda.cvtColor(gpu, cv.COLOR_BGR2GRAY)
        return gpu.download()
    
    def _gpu_blur(self, frame, k=2):
        gpu = self._gpu_upload(frame)
        gpu = cv.cuda.GaussianBlur(gpu, (k, k), 0)
        return gpu.download()
    
    def _gpu_edges(self, frame, t1=80, t2=160):
        gpu = self._gpu_upload(frame)
        gpu = cv.cuda.Canny(gpu, t1, t2)
        return gpu.download()
    

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
        self.gpu_ok = enable
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
        return cpu_t, None
    
    processor.enable_gpu(True)
    t0 = time.time()
    for _ in range(n):
        processor.gray(frame)
        processor.blur(frame)
        processor.edges(frame)
    gpu_t = (time.time() - t0) / n

    logger.info(f"Benchmark: CPU={cpu_t*1000:.2f}ms | GPU={gpu_t*1000:.2f}ms")

    return {"cpu_ms": cpu_t * 1000,
            "gpu_ms": None if gpu_t is None else gpu_t * 1000,
            "gpu_faster": gpu_t is not None and gpu_t < cpu_t}


# from process import benchmark_processor
# from processor import Processor

# processor = Processor(use_gpu=True)

# cpu_t, gpu_t = benchmark_processor(processor, frame)

# if gpu_t is not None and gpu_t < cpu_t * 0.8:
#     processor.enable_gpu(True)
# else:
#     processor.enable_gpu(False)
