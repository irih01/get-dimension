import cv2 as cv
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .processor import Processor
from .measure import MeasureMath, MeasureView
from .contours import Contour
from .detect_aruco import DetectAruco
from .color import get_limits


GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX


class ObjDetector:
    """
    Advanced object detector + measurer.

    Principii:
      - Detectează obiectul geometric (independent de ArUco)
      - Masoara in mm daca exista scala
      - Deseneaza bbox + dimensiuni
    """
    
    def __init__(self, 
                 use_aruco: bool = True, 
                 marker_size: float = 52.0,  
                 draw: bool = True,
                 processor: Processor | None = None):
        """
        :param aruco_enabled: dacă folosim DetectAruco pentru mm_per_px
        :param manual_mm_per_px: fallback mm_per_px (ex: din scriptul tău .npz)
        :param min_area: filtru aria in pixeli
        :param draw: desenează pe frame (bbox, axe, etichete)
        :param precise_mode: dacă True folosește PCA pentru măsurători suplimentare
        :param max_objects: limită pentru numărul de obiecte detectate (None = nelimitat)
        """
       
        self.draw = draw
        self.use_aruco = use_aruco
        self.marker_size = marker_size

        self.processor = processor if processor else Processor(use_gpu=False)

        self.aruco = DetectAruco(marker_size=marker_size, strict=False) if use_aruco else None
        self.mm_per_px = None

        

    # ------------------------------
    # UPDATE SCALE
    # ------------------------------
    def _update_mm_per_px(self, frame: np.ndarray) -> Optional[float]:
        if not self.use_aruco or self.aruco is None:
            self.mm_per_px = None
            return None
        
        # FIX DEFINITIV: Nu mai facem media coeficienților de distorsiune ai camerei!
        # Apelăm metoda dedicată din wrapper care calculează mm_per_px real pe baza pixelilor markerului.
        aruco_data = self.aruco.ret_dimensions(frame)
        if not aruco_data:
            self.mm_per_px = None
            return None
            
        self.mm_per_px = float(aruco_data["mm_per_px"])
        return self.mm_per_px
        
       


    def _detect_object_contour(self, frame):
        cnts, mask, mode, score = Contour.auto_contours(frame, self.processor)

        if not cnts or len(cnts) < 2:
            return None, None, mask
        
        # valid_cnts = []
        # for c in cnts:
        #     area = cv.contourArea(c)
        #     if area < 500:
        #         continue

        #     hull = cv.convexHull(c)
        #     hull_area = cv.contourArea(hull)
        #     solidity = float(area) / hull_area if hull_area > 0 else 0

        #     if solidity > 0.5:
        #         valid_cnts.append(c)
        
        # valid_cnts = sorted(valid_cnts, key=cv.contourArea, reverse=True)

        # if len(valid_cnts) < 2:
        #     return (valid_cnts[0] if len(valid_cnts) > 0 else None), None, mask
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)

        a4_cnt = cnts[0]
        obj_cnt = cnts[1]

        return a4_cnt, obj_cnt, mask
        # return valid_cnts[0], valid_cnts[1], mask
        
    


    # ------------------------------
    # MEASURE OBJECT
    # ------------------------------  
    def _measure_object(self, contour: np.ndarray, mm_per_px) -> Dict[str, Any]:
        rect = cv.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        if w < h:
            w, h = h, w

        meas = MeasureMath(mm_per_px)
        return {"center_px": (float(cx), float(cy)),
                "bbox":rect,
                "width_px": float(w), 
                "height_px": float(h), 
                "width_mm": meas.px_to_mm(w),
                "height_mm": meas.px_to_mm(h),
                "angle": float(angle)}





    def _passes_color_hint(self, frame, contour, min_ratio=0.05):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower, upper = get_limits([0, 0, 255])  # rosu
        mask = cv.inRange(hsv, lower, upper)

        temp = np.zeros(mask.shape, np.uint8)
        cv.drawContours(temp, [contour], -1, 255, -1)

        overlap = cv.bitwise_and(mask, temp)
        ratio = cv.countNonZero(overlap) / cv.contourArea(contour)

        return ratio > min_ratio




    # ----------------------------------------------------------------------
    # Desenare informatii pe frame
    # ----------------------------------------------------------------------
    def _draw(self, frame, a4_cnt, obj_data) -> None:
        cv.drawContours(frame, [a4_cnt], -1, (0, 200, 0), 2)


        box = cv.boxPoints(obj_data["bbox"]).astype(np.int32)
        cv.polylines(frame, [box], True, (0, 0, 255), 2)

        cx, cy = map(int, obj_data["center_px"])
        p1, p2, p3 = box[0], box[1], box[2]

        # FIX SAFE: Verificăm dacă width_mm există înainte de a forța float() ca să evităm crash-ul!
        if obj_data["width_mm"] is not None:
            w_mm = float(obj_data['width_mm'])
            h_mm = float(obj_data['height_mm'])
            text = f"{w_mm:.2f}mm x {h_mm:.2f}mm"
            color = GREEN

            MeasureView.draw_dim_line(frame, p1, p2, obj_data["width_mm"], unit="mm", color=(0, 255, 0))
            # Și pentru înălțime
            MeasureView.draw_dim_line(frame, p2, p3, obj_data["height_mm"], unit="mm", color=(0, 255, 0))
        else:
            w_px = float(obj_data['width_px'])
            h_px = float(obj_data['height_px'])
            text = f"{w_px:.0f}px x {h_px:.0f}px"
            color = ORANGE
            MeasureView.draw_dim_line(frame, p1, p2, obj_data["width_px"], unit="px", color=(0, 165, 255))
            MeasureView.draw_dim_line(frame, p2, p3, obj_data["height_px"], unit="px", color=(0, 165, 255))

        # cv.putText(frame,
        #            text,
        #            (cx + 10, cy - 10),
        #            FONT,
        #            0.6,
        #            color,
        #            2)




    # ----------------------------------------------------------------------
    # Pipeline principal
    # ----------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> dict[str, Any]:
        result = {
            "ok": False,
            "mm_per_px": None,
            "bbox": None,
            "width_mm": None,
            "height_mm": None,
            "width_px": None,  # FIX: Corectat typo ("widht_px" -> "width_px")
            "height_px": None,
            "reason": None
        }

        # 1. mm_per_px
        mm_per_px = self._update_mm_per_px(frame)
        result["mm_per_px"] = mm_per_px


        # 2. contururi
        a4_cnt, obj_cnt, mask = self._detect_object_contour(frame)
        if obj_cnt is None:
            result["reason"] = "object_not_found"
            return result

        # 3.masurare
        obj = self._measure_object(obj_cnt, mm_per_px)

        result.update({
            "ok": True,
            "bbox": obj["bbox"],
            "width_mm": obj["width_mm"],
            "height_mm": obj["height_mm"],
            "width_px": obj["width_px"],
            "height_px": obj["height_px"]
        })


        # 4. desen
        if self.draw:
            self._draw(frame, a4_cnt, obj)

        return result
