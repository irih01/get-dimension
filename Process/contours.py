import cv2 as cv
import numpy as np
from .processor import Processor



class Contour:
    """
    Functii pentru extragere contururi
    """

    # ===========================================================
    #   Prag automat din histogramă
    # ===========================================================
    @staticmethod
    def auto_threshold(gray: np.ndarray) -> int:
        """Determină automat pragul de fundal alb"""
        #pixeli cei mai luminosi
        p = np.percentile(gray, 90)
        return int(max(180, min(240, p - 15)))
        


    # ===========================================================
    # Mască și preprocesare pentru contururi (threshold / canny)
    # ============================================================
    @staticmethod
    def get_mask(frame, 
                 processor: Processor, 
                 mode: str="threshold", 
                 thresh: int | int=None, 
                 ) -> np.ndarray:
        """Returnează mască binară pentru detecție contururi"""         
        if not isinstance(processor, Processor):
            raise TypeError("processor in get_mask trebuie sa fie instanta Processor()")


        g = processor.gray(frame)
        b = processor.blur(g)
        kernel = np.ones((5, 5), np.uint8)

        if mode =="threshold":
            if thresh is None:
                thresh = Contour.auto_threshold(b)
            
            _, mask = cv.threshold(b, thresh, 255, cv.THRESH_BINARY_INV)

        elif mode == "canny":
            low, high = processor.canny_thr
            edges = processor.edges(b, low, high)
            mask = cv.dilate(edges, kernel, iterations=2)

        else:
            raise ValueError("Modul trebuie sa fie 'threshold' sau 'canny'")
        
        # zgomot
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
        return mask
    


    # =========================================================================
    # Extragere contururi
    # =========================================================================
    @staticmethod
    def get_all_cnts(frame, processor: Processor, 
                     mode: str="threshold", 
                     thresh: int | None = None):
        """Returneaza toate contururile"""

        if not isinstance(processor, Processor):
            raise TypeError("processor trebuie sa fie instanta Processor()")

        mask = Contour.get_mask(frame, processor, mode, thresh)
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return cnts, mask
    

    @staticmethod
    def get_largest_cnt(frame, processor: Processor,
                         mode: str = "threshold", 
                         thresh: int | None = None):
        """
        Returneaza cel mai mare contur din imagine.
        """
        if not isinstance(processor, Processor):
            raise TypeError("processor trebuie sa fie instanta Processor()")


        mask = Contour.get_mask(frame, processor, mode, thresh)
        cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not cnts: return None
        return max(cnts, key=cv.contourArea)


    @staticmethod
    def get_filtered_contours(frame,
                              processor: Processor, 
                              min_area: int = 1000, 
                              vertex_filter: int = 0, 
                              mode: str = "threshold", 
                              thresh: int | None = None):
        """
        Returneaza contururi filtrat
        
        :param frame: frame
        :param procesor: modulul procesorului de imagine (cpu/gpu)
        :type procesor: Processor
        :param min_area: Delimitator al ariei minime
        :type min_area: int
        :param vertex_filter: Valoare filtru (defalut 0)
        :type vertex_filter: int
        :param mode: Mod pentru detectie margini/contururi (threshold sau canny)
        :type mode: str
        :param thresh: Valoare prag pentru threshold
        :type thresh: int | None
        """
        cnts, mask = Contour.get_all_cnts(frame, processor, mode, thresh)

        filtered = []
        for c in cnts:
            area = cv.contourArea(c)
            if area < min_area:
                continue

            perimeter = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02*perimeter, True)

            bbox = cv.boundingRect(approx)
            if vertex_filter > 0 and len(approx) != vertex_filter:
                continue
    
            filtered.append({"area": area, 
                             "vertices": len(approx),
                             "approx": approx,
                             "contour": c})

        return sorted(filtered, key = lambda x:x["area"], reverse=True), mask


    @staticmethod
    def auto_contours(frame, processor):
        h, w = frame.shape[:2]
        frame_area = h * w

        cnts_t, mask_t = Contour.get_all_cnts(
            frame, processor, mode="threshold"
        )
        score_t = segmentation_score(cnts_t, frame_area)

        if score_t > 0.7:
            return cnts_t, mask_t, "threshold", score_t

        cnts_c, mask_c = Contour.get_all_cnts(
            frame, processor, mode="canny"
        )
        score_c = segmentation_score(cnts_c, frame_area)

        if score_c > score_t:
            return cnts_c, mask_c, "canny", score_c

        return cnts_t, mask_t, "threshold", score_t




def contours_debug(frame, cnts, mask=None, title="contours"):
    """ 
    processor = Processor(use_gpu=True)
    cnts, mask = Contour.get_all_cnts(frame, processor, mode='threshold')
    
    if debug:
        debug_contours(frame, cnts, mask)"""
    
    dbg = frame.copy()

    for c in cnts:
        cv.drawContours(dbg, [c], -1, (0, 255, 0), 2)

    if mask is not None:
        mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        dbg = np.hstack((dbg, mask_bgr))
    
    cv.imshow(title, dbg)
    cv.waitKey(1)



def segmentation_score(contours, frame_area):
    """Scor de calitate al segmentării.
        Heuristica:
        aria maxima ~= A4
        aria obiectului E [2%, 30%] din A4
        """
    if not contours:
        return 0.0
    
    areas = sorted([cv.contourArea(c) for c in contours], reverse=True)
    a4 = areas[0]

    if a4 < frame_area * 0.1:
        return 0.1
    
    score = 1.0

    if len(areas) > 6:
        score *= 0.6
    
    if len(areas) > 1:
        obj_ratio = areas[1] / a4
        if not (0.02 < obj_ratio < 0.3):
            score *= 0.5
    
    return score




def detect_a4_corners(a4_cnt):
    peri = cv.arcLength(a4_cnt, True)
    approx = cv.approxPolyDP(a4_cnt, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)

    rect = cv.minAreaRect(a4_cnt)
    box = cv.boxPoints(rect)
    return np.int0(box)



def debug_full(frame, cnts, mask, mode, score, a4_corners=None):
    vis = frame.copy()

    for c in cnts:
        cv.drawContours(vis, [c], -1, (0, 255, 0), 2)

    if a4_corners is not None:
        for p in a4_corners:
            cv.circle(vis, tuple(p), 6, (255, 0 ,0), -1)

    cv.putText(vis, f"{mode} | score = {score:.2f}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    out = np.hstack((vis, mask_bgr))

    cv.imshow("Debug", out)
    cv.waitKey(1)