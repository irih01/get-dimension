import cv2 as cv
import numpy as np
import logging
import os
import time

logger = logging.getLogger(__name__)

def sharpness_measure(gray):
    return cv.Laplacian(gray, cv.CV_64F).var()

class ChessboardImageSaver:
    """ Capturarea si salvarea imaginilor de calibrare si detectarea colturilor"""
    def __init__(self, camera_stream, board_dim = (9, 6), save_path: str = 'calibration-images2', criteria = None, min_sharpness=30.0):
        self.cam = camera_stream
        self.board_dim = board_dim

        base_dir = os.getcwd()
        self.save_path = os.path.join(base_dir, save_path)
        
        self.criteria = criteria or (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.min_sharpness = min_sharpness
        
        self.image_count = 0
        self.clicked = False
        self._prepare_directory()

    # --------------------------------- DIRECTORY HELPER ------------------------------
     
    def _prepare_directory(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            logger.info(f"Folder creat: {self.save_path}")
        else:
            logger.info(f" Folderul {self.save_path} exista deja")


    # --------------------------------- DETECT FUNCTIONS ------------------------------

    
    def detect_chessboard(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE

        ret, corners = cv.findChessboardCorners(gray, self.board_dim, flags)

        corners2 = None
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.criteria)
            note = f"found: {ret}, sharp: {sharpness_measure(gray):.2f}"
            frame = cv.drawChessboardCorners(frame, self.board_dim, corners2, ret)
            return frame, True, corners2
        return frame, False, None
        
    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            self.clicked = True

    def start_capture(self):
        logger.info("Pornire captura video.. (apasa 'S' pentru salvare, 'Q' pentru inchidere)")
        cv.namedWindow("Camera")
        cv.setMouseCallback("Camera", self._on_mouse)
        while True:
            ret, frame = self.cam.stream_get_frame()
            if frame is None: 
                time.sleep(0.001) # FIX: Salvăm CPU-ul de la burnout
                continue

            display_frame = frame.copy()
            
            # FIX IMPORTANT: Detectăm colțurile pe cadrul curat, NU după ce desenăm liniile de ghidaj
            # care pot induce în eroare detectorul OpenCV!
            display_frame, found, _ = self.detect_chessboard(display_frame)
            display_frame = self._draw_guide_grid(display_frame)

            cv.putText(display_frame,
                       f'Imagine curenta: {self.image_count}',
                       (30, 40),
                       cv.FONT_HERSHEY_TRIPLEX,
                       1.2,
                       (0, 255, 0),
                       2,
                       cv.LINE_AA)
            
            cv.imshow("Camera", display_frame)

            key = cv.waitKey(1)
            if key == ord('q'):
                break

            if (self.clicked or key == ord('s')) and found:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                sharp = sharpness_measure(gray)
                if sharp < self.min_sharpness:
                    print(f"[Avertisment] Imagine prea blurata (sharp = {sharp:.1f} < {self.min_sharpness}) - skip salvare!")
                    self.clicked = False
                    continue
                    
                filename = os.path.join(self.save_path, f"image{self.image_count}.png")
                cv.imwrite(filename, frame)
                print(f"[INFO] Imagine salvata: {filename}")
                self.image_count += 1
                self.clicked = False
            
        self.cam.stop()
        cv.destroyAllWindows()
        print(f"[INFO] Total imagini salvate: {self.image_count}")

    def _draw_guide_grid(self, frame):
        h, w = frame.shape[:2]
        # Linii verticale
        cv.line(frame, (w//3, 0), (w//3, h), (200, 200, 200), 1)
        cv.line(frame, (2*w//3, 0), (2*w//3, h), (200, 200, 200), 1)
        # Linii orizontale
        cv.line(frame, (0, h//3), (w, h//3), (200, 200, 200), 1)
        cv.line(frame, (0, 2*h//3), (w, 2*h//3), (200, 200, 200), 1)
        
        # Overlay text pentru instructiuni
        cv.putText(frame, "Umple fiecare zona din grid", (10, h - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame



class CircleImageSaver:
    """Captura si salvare imagini dot-grid 11x4 asymetric circle"""
    def __init__(self, camera_stream, pattern_dim=(11, 4), save_path='calibration-images', criteria=None, asymetric=True, min_sharpness=30.0):
        self.cam = camera_stream
        self.pattern_dim = pattern_dim
        self.save_path = os.path.join(os.getcwd(), save_path)
        self.asymetric=  asymetric

        self.criteria = criteria or (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                                     30, 0.1)
                                     
        # FIX: Definim proprietatea min_sharpness care lipsea din constructorul original
        self.min_sharpness = min_sharpness

        self.image_count = 0
        self.clicked = False
        self._prepare_directory()

    def _prepare_directory(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            logger.info(f"Folder creat: {self.save_path}")
        else:
            logger.info(f" Folderul {self.save_path} exista deja")

    
    # Detect
    def detect_circles(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        flags = cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING if self.asymetric else cv.CALIB_CB_SYMMETRIC_GRID
        ret, centers = cv.findCirclesGrid(gray, self.pattern_dim, flags=flags)

        if ret:
            cv.cornerSubPix(gray, centers, (5, 5), (-1, -1), self.criteria)
            frame = cv.drawChessboardCorners(frame, self.pattern_dim, centers, ret)
            return frame, ret, centers

        return frame, False, None

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            self.clicked = True

    # Start captur
    def start_capture(self):
        print("Pornire captura circle grid")
        cv.namedWindow("Camera")
        cv.setMouseCallback("Camera", self._on_mouse)

        while True:
            ret, frame = self.cam.stream_get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            display_frame, found, _ = self.detect_circles(frame.copy())

            text = f"Imagine curenta: {self.image_count}"
            cv.putText(display_frame, 
                       text, 
                       (30, 40), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)

            cv.imshow("Camera", display_frame)
            key = cv.waitKey(1)

            if key == ord('q'):
                break

            if (self.clicked or key == ord('s')) and found:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                sharp = sharpness_measure(gray)
                # FIX: Corectat denumirea proprietății (era typo sharoness)
                if sharp < self.min_sharpness:
                    print(f"Imagine too soft (sharp = {sharp:.1f} < {self.min_sharpness}) - skip")
                    self.clicked = False
                    continue
                filename = os.path.join(self.save_path, f"circle_{self.image_count}.png")
                cv.imwrite(filename, frame)
                print(f"Imagine salvata: {filename}")
                self.image_count += 1
                self.clicked = False
        
        self.cam.stop()
        cv.destroyAllWindows()
        print(f"Total imagini salvate: {self.image_count}")
