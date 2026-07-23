import cv2 as cv
import numpy as np
import os
import time

class ManualDistance:
    def __init__(self, camera_stream, save_path = "manual_input.npz", calib_matrix = "MultiMatrix.npz"):
        self.cam = camera_stream
        self.save_path = save_path
        # FIX: Corectat punctul rătăcit din denumirea implicită
        self.calib_matrix = calib_matrix

        self.points = []
        self.frame = None

        self.mm_p_px = None
        self.px_p_mm = None
        self.focal_length_mm = None
        self.focal_length_px = None

        self._load_intrinsics()

    def _load_intrinsics(self):
        if not os.path.exists(self.calib_matrix):
            self.focal_length_px = None
            return
        
        data = np.load(self.calib_matrix, allow_pickle=True)
        cam_mtx = data["camMatrix"]

        fx = cam_mtx[0, 0]
        fy = cam_mtx[1, 1]
        self.focal_length_px = (fx + fy) / 2

    def _mouse_event(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Lăsăm utilizatorul să pună puncte doar în faza inițială
            if len(self.points) < 2:
                self.points.append((x, y))

    def _draw_ui(self, frame, text):
        y0 = 30
        for i, line in enumerate(text.split("\n")):
            cv.putText(frame, line, (10, y0 + i * 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def run(self):
        cv.namedWindow("Manual Distance")
        cv.setMouseCallback("Manual Distance", self._mouse_event)

        step = "select_points"
        real_dist_mm = None

        while True:
            # FIX: Compatibilitate cu interfața ta unificată de stream
            if hasattr(self.cam, "stream_get_frame"):
                ret, frame = self.cam.stream_get_frame()
            else:
                ret, frame = self.cam.get_frame()
                
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            display = frame.copy()

            # Desenăm permanent cercurile pentru feedback vizual
            for p in self.points:
                cv.circle(display, p, 6, (0, 0, 255), -1)

            # 1.Select pcts
            if step == "select_points":
                ui = " Selecteaza 2 pcte pe obiect \n Click stanga selecteaza"
                ui += f"\nPuncte selectate: {len(self.points)}/2"

                if len(self.points) == 2:
                    step = "ask_distance"

            # 2.Focal point
            elif step == "ask_distance":
                pt1, pt2 = np.array(self.points[0]), np.array(self.points[1])
                px_dist = np.linalg.norm(pt2 - pt1)

                ui = f"Distanta pixel: {px_dist:.1f}px\n"
                ui += "Introdu distanta reala (mm) in consola..."

                # FIX IMPORTANT: Împrospătăm bufferul ferestrei înainte de blocarea input-ului,
                # altfel OpenCV va afișa o fereastră albă și blocată în timp ce utilizatorul tastează!
                cv.imshow("Manual Distance", self._draw_ui(display, ui))
                cv.waitKey(50)

                # Preluăm valoarea din consolă
                try:
                    real_dist_mm = float(input("\nIntrodu distanta in mm: "))
                    self.px_p_mm = px_dist / real_dist_mm
                    self.mm_p_px = 1 / self.px_p_mm
                    step = "compute_focal"
                except ValueError:
                    print("[EROARE] Te rog introdu un număr valid, idiotule!")
                    self.points = []  # Resetăm punctele ca să reincerci
                    step = "select_points"

            # 3. Focal lenght
            elif step == "compute_focal":
                pt1, pt2 = np.array(self.points[0]), np.array(self.points[1])
                px_dist = np.linalg.norm(pt2 - pt1)

                if self.focal_length_px:
                    self.focal_length_mm = self.focal_length_px * self.mm_p_px
                    step = "done"
                else:
                    ui = "Cunosti distanta camera-plan (mm)?\n(y/n) tasteaza in cons.."
                    cv.imshow("Manual Distance", self._draw_ui(display, ui))
                    cv.waitKey(50)
                    
                    choice = input("Cunosti distanta camera-plan (mm)? (y/n): ").strip().lower()

                    if choice == "y":
                        D0 = float(input("Introdu distanta camera-plan (mm): "))
                        self.focal_length_mm = (px_dist * D0) / real_dist_mm
                        self.focal_length_px = self.focal_length_mm / self.mm_p_px
                    else:
                        self.focal_length_mm = None
                        self.focal_length_px = None

                    step = "done"
            
            elif step == "done":
                ui = f"Calibrare finalizata!\n"
                ui += f"{self.px_p_mm:.3f} px/mm\n{self.mm_p_px:.5f} mm/px\n"

                if self.focal_length_px:
                    ui += f"Focala: {self.focal_length_px:.1f}px"
                else:
                    ui += "Focala: necunoscuta"

                display = self._draw_ui(display, ui)
            
            cv.imshow("Manual Distance", display)
            key = cv.waitKey(1)

            if step == "done" and key == ord('q'):
                self.save()
                break
                
        cv.destroyAllWindows()

    def save(self):
        # FIX: Compatibilitate cu interfața unificată la salvare
        if hasattr(self.cam, "stream_get_frame"):
            ret, frame = self.cam.stream_get_frame()
        else:
            ret, frame = self.cam.get_frame()
            
        if not ret or frame is None:
            print("[ERROR] Nu s-a putut prelua cadrul pentru salvare.")
            return

        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        fx = fy = self.focal_length_px if self.focal_length_px else 500.0 # Valoare standard de siguranță

        manual_cam_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0,  1]
        ], dtype=np.float32)

        manual_dist_coef = np.zeros((5, 1), dtype=np.float32)

        np.savez(self.save_path,
                 camMatrix = manual_cam_matrix,
                 distCoef = manual_dist_coef, 
                 px_per_mm = self.px_p_mm,
                 mm_per_pixel = self.mm_p_px,
                 focal_len_mm = self.focal_length_mm,
                 focal_len_px = self.focal_length_px,
                 points = np.array(self.points))
        
        print(f"[SUCCES] Datele manuale au fost salvate în {self.save_path}")
        
    @staticmethod
    def load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        data = np.load(path, allow_pickle=True)
        # FIX DEFINITIV: Am corectat denumirea cheilor ca să se potrivească la fix cu ce salvezi tu în save()!
        return {
            "pixels_per_mm": float(data["px_per_mm"]),
            "mm_per_pixel": float(data["mm_per_pixel"]),
            "focal_length_mm": float(data["focal_len_mm"]) if "focal_len_mm" in data and data["focal_len_mm"] is not None else None,
            "focal_length_px": float(data["focal_len_px"]) if "focal_len_px" in data and data["focal_len_px"] is not None else None,
            "points": data["points"]
        }
