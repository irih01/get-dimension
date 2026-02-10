import cv2 as cv
import numpy as np
import os

class ManualDistance:
    def __init__(self, camera_stream, save_path = "manual_input.npz", calib_matrix = ".MultiMatrix.npz"):
        self.cam = camera_stream
        self.save_path = save_path
        self.calib_matrix= calib_matrix

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
            ret, frame = self.cam.get_frame()
            if not ret or frame is None:
                continue

            display = frame.copy()

            # 1.Select pcts
            if step == "select_points":
                ui = " Selecteaza 2 pcte pe obiect \n Click stanga selecteaza"
                ui += f"\nPuncte selectate: {len(self.points)}/2"

                for p in self.points:
                    cv.circle(display, p, 6, (0, 0, 255), -1)

                if len(self.points) == 2:
                    step = "ask_distance"

            # 2.Focal point
            elif step == "ask_distance":
                pt1, pt2 = np.array(self.points[0]), np.array(self.points[1])
                px_dist = np.linalg.norm(pt2 - pt1)

                ui = f"Distanta pixel: {px_dist:.1f}px\n"
                ui += "Introdu distanta reala (mm) in consola..."

                cv.imshow("Manual Distance", self._draw_ui(display, ui))
                cv.waitKey(1)

                # valoare reala
                real_dist_mm = float(input("\nIntrodu distanta in mm: "))
                self.px_p_mm = px_dist / real_dist_mm
                self.mm_p_px = 1 / self.px_p_mm

                step = "compute_focal"

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
                    cv.waitKey(1)
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
        ret, frame = self.cam.get_frame()
        if not ret or frame is None:
            print("[ERROR] Nu s-a putut prelua cadrul pentru salvare.")
            return

        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        fx = fy = self.focal_length_px

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
                 mm_per_pixel=self.mm_p_px,
                 focal_len_mm = self.focal_length_mm,
                 focal_len_px = self.focal_length_px,
                 points = np.array(self.points))
        
        print(f"[SUCCES]")
        
    @staticmethod
    def load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        data = np.load(path, allow_pickle=True)
        return {
            "pixels_per_mm": float(data["pixels_per_mm"]),
            "mm_per_pixel": float(data["mm_per_pixel"]),
            "focal_length_mm": float(data["focal_length_mm"])
                if "focal_length_mm" in data else None,
            "focal_length_px": float(data["focal_length_px"])
                if "focal_length_px" in data else None,
            "points": data["points"]
        }