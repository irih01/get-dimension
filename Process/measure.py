import cv2 as cv
import numpy as np
import math


class MeasureMath:
    def __init__(self, mm_per_px = None):
        self.mm_per_px = float(mm_per_px) if mm_per_px is not None else None

    def px_to_mm(self, px):
        if self.mm_per_px is None:
            return None
        return float(px * self.mm_per_px)
    
    def mm_to_px(self, mm):
        if self.mm_per_px is None:
            return None
        return mm / self.mm_per_px
    
    def distance(self, p1, p2):
        p1 = np.array(p1, float)
        p2 = np.array(p2, float)

        px = float(np.linalg.norm(p1 - p2))
        mm = self.px_to_mm(px)

        return {"px": px, "mm": mm}
    
    def box_dim(self, x, y, w: float, h: float):
        w_px = float(w)
        h_px = float(h)
        w_mm = self.px_to_mm(w_px)
        h_mm = self.px_to_mm(h_px)

        return {
            "width_px": w_px,
            "height_px": h_px,
            "width_mm": w_mm,
            "height_mm": h_mm
        }
    
    def area(self, contour):
        px2 = float(cv.contourArea(contour))
        mm2 = self.px_to_mm(1)**2 * px2 if self.mm_per_px else None
        return {"px2": px2, "mm2": mm2}
    
    def calculate_a4(self, marker_corners, known_marker_size, a4_size = (21.0, 29.7)):
        marker_corners = marker_corners.reshape(4, 2)
        pixel_w = np.linalg.norm(marker_corners[0] - marker_corners[1])
        pixel_h = np.linalg.norm(marker_corners[1] - marker_corners[2])
        avg_mark_px = (pixel_w + pixel_h) / 2

        px_per_cm = avg_mark_px / known_marker_size
        cm_per_px = 1 / px_per_cm

        a4_w_px = a4_size[0] * px_per_cm
        a4_h_px = a4_size[1] * px_per_cm

        return {
            "px_per_cm": px_per_cm,
            "cm_per_px": cm_per_px,
            "a4_w_px": a4_w_px,
            "a4_h_px": a4_h_px
        }



class MeasureView:
    @staticmethod
    def draw_dim_line(frame, p1, p2, val, color=(0, 255, 0), thickness=2, font_size=0.7, unit="mm", offset=20):
        pt1 = tuple(map(int, p1))
        pt2 = tuple(map(int, p2))

        # Vector driectie si lungime
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return frame
        
        # Vecor perpendicular
        perp = np.array([-dy, dx], dtype=float)
        perp /= np.linalg.norm(perp)
        offset_vec = perp * offset

        # Pcte pentru linia de cota
        p1_off = (int(pt1[0] + offset_vec[0]), int(pt1[1] + offset_vec[1]))
        p2_off = (int(pt2[0] + offset_vec[0]), int(pt2[1] + offset_vec[1]))

        cv.line(frame, p1_off, p2_off, color, thickness, cv.LINE_AA)

        # Sageti
        arrw_len = 10
        cv.arrowedLine(frame, p1_off, p2_off, color, thickness, tipLength=0.03)
        cv.arrowedLine(frame, p2_off, p1_off, color, thickness, tipLength=0.03)

        # Text
        mid = ((p1_off[0] + p2_off[0]) // 2, (p1_off[1] + p2_off[1]) // 2)
        #text = f'{val:.1f} {unit}'
        
        (tw, th), _ = cv.getTextSize(f'{val:.1f} {unit}', cv.FONT_HERSHEY_SIMPLEX, font_size, 1)
        txt_pos = (int(mid[0] - tw // 2), int(mid[1] - offset_vec[1] // 2))

        # Fundal alb
        cv.rectangle(frame, (txt_pos[0] - 4, txt_pos[1] - th - 4), 
                    (txt_pos[0] + tw + 4, txt_pos[1] + 4), (255, 255, 255), -1)


        # Text propriu-zis
        cv.putText(frame, f'{val:.1f} {unit}', txt_pos, cv.FONT_HERSHEY_SIMPLEX, font_size, color, 1, cv.LINE_AA)

        return frame
    
    @staticmethod
    def draw_distance(frame, p1, p2, info):
        cv.line(frame, p1, p2, (0, 255, 0), 2)
        mid = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

        txt_px = f"{info['px']:.1f}px"
        txt_mm = f"{info['mm']:.1f}mm" if info["mm"] else None
        txt = txt_px if txt_mm is None else f"{txt_px} / {txt_mm}"

        cv.putText(frame, txt, mid, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
    
    
    @staticmethod
    def draw_box(frame, x, y, w, h, info):
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)

        txt_px = f"{info['width_px']:.1f}px x {info['height_px']:.1f}px"
        txt_mm = ""
        if info['width_mm'] is not None:
            txt_mm = f" | {info['width_mm']:.1f}mm x {info['height_mm']:.1f}mm"

        cv.putText(frame, txt_px + txt_mm, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        return frame
    





class Measure:
    """
    Modul pentru masuratori in imagine folosind un factor de scala mm per px.
    Poate fi utilizat cu orice sistem de calibrare (ex: ArUco).
    """

    def __init__(self, mm_per_px: float = None):
        """mm_per_px: milimetri per pixel
        Daca e None, functiile vor intoarce valori doar in pixeli.
        """
        self.mm_per_px = mm_per_px

    
    @staticmethod
    def draw_dim_line(frame, pt1, pt2, val, 
                      color=(0, 255, 0), 
                      thickness=2, 
                      font_sz=0.7, 
                      unit="mm",
                      offset=20):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        # Vector driectie si lungime
        dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return frame
        
        # Vecor perpendicular
        perp = np.array([-dy, dx], dtype=float)
        perp /= np.linalg.norm(perp)
        offset_vec = perp * offset

        # Pcte pentru linia de cota
        p1_off = (int(pt1[0] + offset_vec[0]), int(pt1[1] + offset_vec[1]))
        p2_off = (int(pt2[0] + offset_vec[0]), int(pt2[1] + offset_vec[1]))

        cv.line(frame, p1_off, p2_off, color, thickness, cv.LINE_AA)

        # Sageti
        arrw_len = 10
        cv.arrowedLine(frame, p1_off, p2_off, color, thickness, tipLength=0.03)
        cv.arrowedLine(frame, p2_off, p1_off, color, thickness, tipLength=0.03)

        # Text
        mid = ((p1_off[0] + p2_off[0]) // 2, (p1_off[1] + p2_off[1]) // 2)

        if isinstance(val, (int, float, np.floating)):
            text = f'{val:.1f} {unit}'
        else:
            text = f'{val}'
        
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_sz, 1)
        txt_pos = (int(mid[0] - tw // 2), int(mid[1] - offset_vec[1] // 2))

        # Fundal alb
        cv.rectangle(frame, (txt_pos[0] - 4, txt_pos[1] - th - 4), 
                    (txt_pos[0] + tw + 4, txt_pos[1] + 4), (255, 255, 255), -1)


        # Text propriu-zis
        cv.putText(frame, text, txt_pos, cv.FONT_HERSHEY_SIMPLEX, font_sz, color, 1, cv.LINE_AA)

        return frame

   

    
