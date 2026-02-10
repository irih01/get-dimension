import PySpin
import cv2 as cv
import threading
from Errors import CameraError, safe_call, logger

from .base_camera import BaseCamera
from typing import Optional


class FLIRCamera(BaseCamera):
    def __init__(self, 
                 cam_id: int = 0, 
                 image_format: str = "Mono8",
                 fps: float = 60.0,
                 timeout: int = 100, 
                 use_gpu: bool = False, 
                 exposure_mode: str = "auto", 
                 exposure_val: float = 15000.0,
                 wb_mode: str = "auto", 
                 wb_red: float = 1.0, 
                 wb_blue: float = 1.0, 
                 gain_mode: str = "auto", 
                 gain_val: float = 5.0,
                 system: Optional[PySpin.System] = None):
        super().__init__(use_gpu)

        self.cam_idx = cam_id
        self.timeout = timeout
        self.px_format = image_format
        self.fps = fps

        self.exposure_mode = exposure_mode
        self.exposure_val = exposure_val

        self.wb_mode = wb_mode
        self.wb_red = wb_red
        self.wb_blue = wb_blue

        self.gain_mode = gain_mode
        self.gain_val = gain_val

        # PySpin system management
        self._own_system = False
        if system is None:
            self.system = PySpin.System.GetInstance()
            self._own_system = True
        else:
            self.system = system

        # Cameras
        cam_list = self.system.GetCameras()
        if cam_list.GetSize() == 0:
            cam_list.Clear()
            raise RuntimeError("Nu a fost gasita nici o camera FLIR!")
        
        if self.cam_idx >= len(cam_list):
            raise IndexError(f"Camera index {cam_id} nu exista (size = {self.cam_list.GetSize()}).")
        
        #self.cam = cam_list[cam_id]
        self.cam = cam_list.GetByIndex(self.cam_idx)
        self.cam_list = cam_list

        self._configured = False
   
    

    # --------------------------------- CORE CONFIG -------------------------------
    def _config(self):

        if self._configured:
            logger.info("Camera deja configurata.")
            return
        try:
            self.cam.Init()
            nodemap = self.cam.GetNodeMap()
            stream = self.cam.GetTLStreamNodeMap()
        except Exception as e:
            logger.error(f"Camera init error: {e}")
            raise

        
        self.set_frame_rate(self.fps)
        self.set_px_format(nodemap)
        self._set_stream_buffer(stream)
        self._set_acquisition_mode(nodemap)
        self._set_exp(nodemap)
        self._set_gain(nodemap)
        self._set_white_balance(nodemap)

        self._configured = True
        logger.info("Camera FLIR initializata")

   

    def start(self):
        self.running = True
        self._config()
        self.thread = threading.Thread(target=self._thread_loop, daemon=True)
        self.thread.start()
        logger.info("Start thread achiziție")
    

    def read_raw(self):
        """try:
            img = self.cam.GetNextImage(self.timeout)
        except PySpin.SpinnakerException as ex:
            logger.error(f"[FLIR] GetNextImage exception: {ex}")
            return None
        except Exception as ex:
            logger.error(f"[FLIR] Unexpected GetNextImage error: {ex}")
            return None

        try:
            if img.IsIncomplete():
                logger.warning(f"[FLIR] Incomplete image (status={img.GetImageStatus()}).")
                return None

            nd = img.GetNDArray()
            if nd is None:
                logger.warning("[FLIR] GetNDArray returned None.")
                return None

            # Convert to 3-channel BGR so downstream processing is consistent
            try:
                if nd.ndim == 2:
                    # mono or raw bayer: decide based on requested pixel format
                    if str(self.px_format).lower().startswith("mono"):
                        bgr = cv.cvtColor(nd, cv.COLOR_GRAY2BGR)
                    else:
                        # attempt a common Bayer conversion; if fails, fallback to gray2bgr
                        try:
                            # choose a likely Bayer → BGR conversion (many cameras use RG or BG)
                            bgr = cv.cvtColor(nd, cv.COLOR_BAYER_RG2BGR)
                        except Exception:
                            try:
                                bgr = cv.cvtColor(nd, cv.COLOR_BAYER_BG2BGR)
                            except Exception:
                                bgr = cv.cvtColor(nd, cv.COLOR_GRAY2BGR)
                elif nd.ndim == 3:
                    # assume already color (BGR) - keep as-is
                    bgr = nd
                else:
                    logger.warning(f"[FLIR] Unexpected NDArray shape: {nd.shape}")
                    return None

                return bgr

            except Exception as ex:
                logger.error(f"[FLIR] Error converting NDArray to BGR: {ex}")
                return None

        finally:
            try:
                if img is not None:
                    img.Release()
            except Exception:
                pass"""
        if not self.running:
            return None
        
        img = None
        try:
            img = self.cam.GetNextImage(self.timeout)  #timeout ms
        
        
            if img.IsIncomplete():
                logger.warning(f"Frame incomplet: {img.GetImageStatus()}")
                img.Release()
                return None
            
            frame = img.GetNDArray()
            
            if self.px_format == "Mono8":
                return frame
            elif self.px_format == "BayerBG8":
                frame = cv.cvtColor(frame, cv.COLOR_BAYER_RG2BGR)
                
            img.Release()

            return frame
        
        except PySpin.SpinnakerException as ex:
            logger.error(f"PySpin exception: {ex}")
            return None
    

    def stop(self):
        if not self.running:
            try:
                if self.cam is not None:
                    try:
                        self.cam.EndAcquisition()
                    except Exception:
                        pass
                    try:
                        self.cam.DeInit()
                    except Exception:
                        pass
            except Exception:
                pass
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

        try:
            if self.cam is not None:
                try:
                    self.cam.EndAcquisition()
                except Exception:
                    logger.warning("EndAcquisition() a dat greș sau nu rula.")
                
                try:
                    self.cam.DeInit()
                except Exception:
                    logger.warning("DeInit() a dat greși sau camera a fost deja dezinițializată.")
            
            try:
                if self.cam_list is not None:
                    self.cam_list.Clear()
            except Exception:
                pass

            self.cam = None
            self._configured = False
            self.system.ReleaseInstance()
            logger.info("Camera FLIR elberata.")
        except Exception as e:
            logger.warning(f"Eroare la eliberarea camerei FLIR: {e}")




    #----------------------------------------------------------------------------------------------------------
    #                                       CAMERA CONFIG
    #----------------------------------------------------------------------------------------------------------

    def set_px_format(self, node):
        try:
            if not self._safe_set_enum(node, "PixelFormat", self.px_format):
                logger.warning(f"Nu s-a putut seta formatul imagini în {self.px_format}")
            else:
                logger.info(f"Format imagine setat: {self.px_format}")
        except Exception as e:
            logger.warning(f"Eroare setare format imagine: {e}")

    def _set_frame_rate(self, node ,fps: float):
        try:
            try:
                if hasattr(self.cam, "AcquisitinFrameRateEnable") and hasattr(self.cam, "AcquisitionFrameRate"):
                    self._safe_set_bool(node, "AcquisitinFrameRateEnable", True)
                    fps = self._safe_set_float(node, "AcquisitionFrameRate", fps)

                    if fps:
                        logger.info(f"Frame rate setat {fps} fps")
                    else:
                        logger.warning("Nu s-a putut seta rata de achiziție (driver restricționat)")
            except Exception:
                logger.warning(f"Nodurile AcquisitionFrameRate nu sunt utilizabile pe această cameră")
        except Exception as e:
            logger.warning(f"Eroare configurare frame rate: {e}")



    def set_frame_rate(self, fps: float):
        try:
            if hasattr(self.cam, "AcquisitinFrameRateEnable") and hasattr(self.cam, "AcquisitionFrameRate"):
                try:
                    
                    

                    if PySpin.IsAvailable(self.cam.AcquisitionFrameRateEnable) and PySpin.IsWritable(self.cam.AcquisitionFrameRateEnable):
                        self.cam.AcquisitionFrameRateEnable.SetValue(True)
                        
                        if PySpin.IsAvailable(self.cam.AcquisitionFrameRate) and PySpin.IsWritable(self.cam.AcquisitionFrameRate):
                            try:
                                self.cam.AcquisitionFrameRate.SetValue(fps)
                                logger.info(f"Framerate setat: {fps}")
                            except Exception:
                                logger.info("Nu s-a putut seta rata de achiziție (driver restricționat)")
                except Exception:
                    logger.warning(f"Nodurile AcquisitionFrameRate nu sunt utilizabile pe această cameră")
        except Exception as e:
            logger.warning(f"Eroare configurare frame rate: {e}")
    

    def _set_stream_buffer(self, stream):
        try:
            handling_node = PySpin.CEnumerationPtr(stream.GetNode("StreamBufferHandlingMode"))
            if PySpin.IsAvailable(handling_node) and PySpin.IsWritable(handling_node):
                newst_entry = handling_node.GetEntryByName("NewestOnly")
                if PySpin.IsAvailable(newst_entry) and PySpin.IsReadable(newst_entry):
                    handling_node.SetIntValue(newst_entry.GetValue())
                    logger.info("StreamBufferHnadlingMode = NewestOnly")
        except Exception as e:
            logger.warning(f"Nu s-a putut seta stream buffer-ul: {e}")

    

    def _set_acquisition_mode(self, node):
        try:
            ac_mode = PySpin.CEnumerationPtr(node.GetNode("AcquisitionMode"))
            if PySpin.IsAvailable(ac_mode) and PySpin.IsWritable(ac_mode):
                continuous = ac_mode.GetEntryByName("Continuous")
                ac_mode.SetIntValue(continuous.GetValue())

            self.cam.BeginAcquisition()
            logger.info("AcquisitionMode = Continuous, BeginAqcuisition()")
        except Exception as e:
            logger.warning(f"Nu s-a putut seta modul de achiziție: {e}")

# def _set_acquisition_mode(self, nodemap):
#     ok = self._safe_set_enum(nodemap, "AcquisitionMode", "Continuous")

#     if not ok:
#         logger.warning("[FLIR] Could not set AcquisitionMode=Continuous")

#     # begin acquisition (safe)
#     try:
#         self.cam.BeginAcquisition()
#         logger.info("[FLIR] BeginAcquisition()")
#     except Exception:
#         logger.debug("[FLIR] BeginAcquisition() failed (maybe already running)")


# def _set_exposure(self, nodemap):
#     mode = self.exposure_mode.lower()

#     if mode == "auto":
#         self._safe_set_enum(nodemap, "ExposureAuto", "Continuous")

#     elif mode == "once":
#         self._safe_set_enum(nodemap, "ExposureAuto", "Once")

#     elif mode == "manual":
#         if self._safe_set_enum(nodemap, "ExposureAuto", "Off"):
#             self._safe_set_float(nodemap, "ExposureTime", self.exposure_val)

#     logger.info(f"[FLIR] Exposure mode={self.exposure_mode}, val={self.exposure_val}")



# def _set_white_balance(self, nodemap):
#     mode = self.wb_mode.lower()

#     if mode == "auto":
#         self._safe_set_enum(nodemap, "BalanceWhiteAuto", "Continuous")

#     elif mode == "once":
#         self._safe_set_enum(nodemap, "BalanceWhiteAuto", "Once")

#     elif mode == "manual":
#         # set Off + try to set BalanceRatio
#         if self._safe_set_enum(nodemap, "BalanceWhiteAuto", "Off"):
#             self._safe_set_float(nodemap, "BalanceRatio", self.wb_red)

#     logger.info(f"[FLIR] White balance mode={self.wb_mode}, red_ratio={self.wb_red}")

    
    def _set_exp(self, node):
        try:
            exp_node = PySpin.CEnumerationPtr(node.GetNode("ExposureAuto"))
            if not (PySpin.IsAvailable(exp_node) and PySpin.IsWritable(exp_node)):
                return
            
            mode = self.exposure_mode.lower()

            if mode == "auto":
                exp_node.SetIntValue(exp_node.GetEntryByName("Continuous").GetValue())
                logger.info("Exposure = Continuous")
                
            
            elif mode == "once":
                exp_node.SetIntValue(exp_node.GetEntryByName("Once").GetValue())
                logger.info("Exposure = Once ")
                
            
            elif mode == "manual":
                exp_node.SetIntValue(exp_node.GetEntryByName("Off").GetValue())
                self._safe_set_float(node, "ExposureTime", self.exposure_val)
                logger.info(f"Exposure = Manual, ExposureTime = {self.exposure_val} us")

        except Exception as e:
            logger.warning(f"Eroare configurare Expunere: {e}")
    

    
    def _set_gain(self, node):
        try:
            gain_node = PySpin.CEnumerationPtr(node.GetNode("GainAuto"))
            if not (PySpin.IsAvailable(gain_node) and PySpin.IsWritable(gain_node)):
                return
            
            if self.gain_mode.lower() == "auto":
                gain_node.SetIntValue(gain_node.GetEntryByName("Continuous").GetValue())
                logger.info("GainAuto = Continuous")
            else:
                gain_node.SetIntValue(gain_node.GetEntryByName("Off").GetValue())
                self._safe_set_float(node, "Gain", self.gain_val)
                logger.info(f"GainAuto = Off, Gain = {self.gain_val}")
        except Exception as e:
            logger.warning(f"Eroare configurare Câștig: {e}")



    def _set_white_balance(self, node):
        try:
            wb_node = PySpin.CEnumerationPtr(node.GetNode("BalanceWhiteAuto"))
            if not (PySpin.IsAvailable(wb_node) and PySpin.IsWritable(wb_node)):
                return
            
            if self.wb_mode.lower == "auto":
                wb_node.SetIntValue(wb_node.GetEntryByName("Continuous").GetValue())
                logger.info("Balance White Auto = Continuous")

            elif self.wb_mode == "once":
                wb_node.SetIntValue(wb_node.GetEntryByName("Once").GetValue())
                logger.info("Balance White Auto = Once")

            elif self.wb_mode == "manual":
                wb_node.SetIntValue(wb_node.GetEntryByName("Off").GetValue())
                self._safe_set_float(node, "BalanceRatio", self.wb_red)
                logger.info(f"White Balance = Manual, Ratio(R) = {self.wb_red}")

        except Exception as e:
            logger.warning(f"Eroare configurare Balanță de Alb: {e}")






 # ---------------------------- CONFIG HELPER ---------------------------------------

    def _safe_set_enum(self, node_map, node_name: str, entry_name: str):
        try:
            enum_node = PySpin.CEnumerationPtr(node_map.GetNode(node_name))
            if not PySpin.IsAvailable(enum_node) or not PySpin.IsWritable(enum_node):
                logger.debug(f"Enum node '{node_name}' nu e valabil.")
                return False
            
            entry = enum_node.GetEntryByName(entry_name)
            if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
                logger.debug(f"Enum entry {entry_name} nu e valabil")
                return False
            
            enum_node.SetIntValue(entry.GetValue())
            return True
        except Exception as e:
            logger.warning(f"Eroare la setarea enum {node_name} -> {entry_name}: {e}")
            return False
        

    def _safe_set_float(self, node_map, node_name: str, value: float):
        try:
            node = PySpin.CFloatPtr(node_map.GetNode(node_name))
            if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                logger.debug(f"Float node {node_name} nu e valabil")
                return False
            
            # If min/max valabil
            if PySpin.IsAvailable(node.GetMin()) and PySpin.IsAvailable(node.GetMax()):
                vmin = node.GetMin()
                vmax = node.GetMax()
                value = max(vmin, min(vmax, float(value)))
            node.SetValue(float(value))
            return True

        except Exception as e:
            logger.warning(f"Eroare setare float node {node_name} = {value}: {e}")
            return False
        

    def _safe_set_bool(self, node_map, node_name: str, val: bool):
        try:
            node = PySpin.CBooleanPtr(node_map.GetNode(node_name))
            if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                logger.debug(f"[FLIR] Bool node {node_name} nu e valabil")
                return False
            node.SetValue(bool(val))
            return True

        except Exception as e:
            logger.warning(f"[Error setare bool node {node_name} = {val} : {e}")
            return False




    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.stop()