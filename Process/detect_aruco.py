from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from cv2 import aruco
import numpy as np
import cv2 as cv
import os



class ArucoCalibLoader:
    def __init__(self, calib_path: Optional[str] = None):
        if calib_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            default_path = os.path.join(base_dir, "Calibration", "FlirMultiMatrix.npz")

        
        self.calibrated_data_path = calib_path or default_path

       
        if not os.path.exists(self.calibrated_data_path):
            msg = f"[ERROR] Fisierul de calibrare nu exista: {self.calibrated_data_path}"
            #logger.warning(Fisierul de calibrare nu exista: {self.calibrated_data_path})
            self.camera_matrix = None
            self.dist = None
            raise FileNotFoundError(msg)
        else:
            try:
                data = np.load(self.calibrated_data_path, allow_pickle=True)
                self.camera_matrix = data["camMatrix"]
                self.dist = data["distCoef"]
                #logger.info(f"Datele de calibrare au fost incarcate din {self.calibrated_data_path}")    
            except Exception as e:
                print(f"[ERROR]")
                self.camera_matrix = None
                self.dist = None
        

class DetectorAruco:
    def __init__(self, 
                 marker_dict_type = aruco.DICT_4X4_50, 
                 thresh_const = 7, 
                 corner_refine = aruco.CORNER_REFINE_SUBPIX):
        
        self.dict = aruco.getPredefinedDictionary(marker_dict_type)
        
        params = aruco.DetectorParameters()
        params.adaptiveThreshConstant = thresh_const
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.cornerRefinementMethod = corner_refine


        self.detector = aruco.ArucoDetector(self.dict, params)

    def detect(self, gray):
        """Returenaza:
            -corners: list (N, 4, 2)
            -ids: list (N,)"""
        
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids, rejected


class ArucoPose:
    def __init__(self, calibration: ArucoCalibLoader, marker_size_cm: float = 5.0):
        self.cal = calibration
        self.marker_size_cm = marker_size_cm

    def estimate_pose(self, corners, ids):
        if ids is None or len(corners) == 0:
            return None
        
        if self.camera_matrix is None or self.dist is None:
           # logger.warning("Camera calibration missing; pose estimation skipped.")
            return None
        
        # note: aruco.estimatePoseSingleMarkers expects markerSize in same units as tVec (meters)
        rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_size_cm, self.cal.camera_matrix, self.cal.dist
        )

        poses = []
        for i, corner in enumerate(corners):
            marker_corners = corner.reshape((4, 2))
            tl, tr, br, bl = marker_corners

            delta = tr - tl
            angle = np.degrees(np.arctan2(delta[1], delta[0]))

            distance_cm = float(np.linalg.norm(tVecs[i]))
            

            poses.append({
                "id": int(ids[i][0]),
                "corners": marker_corners,
                "rVec": rVecs[i],
                "tVec": tVecs[i],
                "distance_cm": distance_cm,
                "angle": angle
            })
        return poses
    
    def estimate_distance(self, tVec):
        """Calculeaza distanta fatade camera"""
        return np.linalg.norm(tVec[0])
    
    def mm_per_px(self, corners):
        if corners is None or len(corners) == 0:
            return None
        pts = corners[0].reshape((4,2))
        tl, tr, br, bl = pts

        px_w = np.linalg.norm(tr - tl)
        px_h = np.linalg.norm(tl - bl)

        avg_mark_px = (px_w + px_h) / 2.0

        mark_mm = self.marker_size * 10.0
        mm_per_px = mark_mm / avg_mark_px

        delta = tr - tl
        angle = np.degrees(np.arctan2(delta[1], delta[0]))

        return {
            "mm_per_px": mm_per_px,
            "angle": angle,
            "corners": pts
        }
    

class ArucoGeometry:
    @staticmethod
    def calculate_a4(marker_corners, marker_size_cm, a4_size_cm = (21.0, 29.7)):
        """
        Calculeaza dimensiunile unei foi A4 in pixeli.
        
        :param marker_corners: Coltuirile marker-ului
        :param marker_size_cm: Dimensiunea reala a marker-ului (cm)
        :param a4_size_cm: Dimensiunea reala a unei foi A4 (21 x 29.7)
        """
        pts = marker_corners.reshape(4, 2)

        p_w = np.linalg.norm(pts[0] - pts[1])
        p_h = np.linalg.norm(pts[1] - pts[2])
        avg_p = (p_w + p_h) / 2

        p_per_cm = avg_p / marker_size_cm
        cm_per_p = 1.0 / p_per_cm

        a4_w = a4_size_cm[0] * p_per_cm
        a4_h = a4_size_cm[1] * p_per_cm
        return{
            "p_per_cm": p_per_cm,
            "cm_per_p": cm_per_p,
            "a4_w": a4_w,
            "a4_h": a4_h
        }
        



# *****************************************************************************************
# Wrapper
# _________________________________________________________________________________________

class DetectAruco:
    """
    Detecteaza merkere ArUco, estimeaza pozitia, calculeaza distante
    dimensiuni si poate desena infromatiile pe frame (Axe, orientare 3D).
    """
    
    def __init__(self, 
                 calibrated_data_path: Optional[str] = None, 
                 marker_dict_type = aruco.DICT_4X4_50, 
                 marker_size: float = 5.0, 
                 marker_unit: str = "cm",
                 use_gpu: bool = False,
                 strict: bool = True
                 ):
        
        self.loader = ArucoCalibLoader(calibrated_data_path)

        self.camera_matrix = self.loader.camera_matrix
        self.dist = self.loader.dist

        if strict and self.camera_matrix is None:
            raise FileNotFoundError("Calibrarea camerei este obligatorie")

        
        # === marker size in metrii ===
        self.marker_size_m = self._ensure_marker_size_m(marker_size, marker_unit)

        # === aruco detector config
        self.detector = DetectorAruco(marker_dict_type)
        

        # GPU
        self.use_gpu = False
        if use_gpu:
            try:
                if cv.cuda.getCudaEnabledDeviceCount() > 0:
                    self.use_gpu = True
            except Exception:
                pass

     
      
    

    # -----------------------------------------------------------------------
    # Incarcare date calibrare
    # def _load_calibration_data(self):
    #     data = np.load(self.calibrated_data_path, allow_pickle=True)
        
    #     if "camMatrix" not in data or "distCoef" not in data:
    #         raise ValueError(f"File {self.calibrated_data_path} lipses cheile cerute (camMatrix, distCoef)")
        
    #     self.camera_matrix = data["camMatrix"]
    #     self.dist = data["distCoef"]
    #     #logger.info(f"Datele de calibrare au fost incarcate din {self.calibrated_data_path}")    



    def _ensure_marker_size_m(self, size: float, unit: str) -> float:
        unit = unit.lower().strip()
        if unit == "cm":
            return float(size) / 100.0
        elif unit == "m":
            return float(size)
        elif unit == "mm":
            return float(size) / 1000.0
        else:
            raise ValueError(f"Unitate necunoscuta: {unit}. Foloseste 'cm' , 'mm' sau 'm' ")



    # =========================================================================
    #   GPU/CPU Preprocesare
    # =========================================================================
    def _preprocess(self, frame):
        if not self.use_gpu:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.equalizeHist(gray)
            return gray
        
        # GPU
        gpu = cv.cuda.GpuMat()
        gpu.upload(frame)
        gpu_gray = cv.cuda.cvtColor(gpu, cv.COLOR_BGR2GRAY)
        try:
            gpu_gray = cv.cuda.equalizeHist(gpu_gray)
        except Exception:
            pass
        return gpu_gray.download()
    




    # ==========================================================================
    # PROCESARE Detect (2D)
    # ====================================================================
    def detect(self, frame: np.ndarray) -> Tuple[Optional[List[np.ndarray]], 
                                                 Optional[np.ndarray], 
                                                 Optional[List[np.ndarray]]]:
        gray = self._preprocess(frame)
        corners, ids, rejected = self.detector.detect(gray)
        return corners, ids, rejected
    





    # =========================================================================
    # Pose estimation (3D)
    # =========================================================================
    def estimate_pose(self, frame) -> Optional[List[Dict[str, Any]]]:
        """
        Returneaza lista de dictionare cu datele 3D ale markerelor.
        Distantele returnate sunt in CM.
        """


        if self.camera_matrix is None or self.dist is None:
           # logger.warning("Camera calibration missing; pose estimation skipped.")
            return None
     
        corners, ids, _ = self.detect(frame)
        if ids is None or len(corners) == 0:
            return None
        


        # IMPORTANT:
        # aruco.estimatePoseSingleMarkers primeste markerSize in metri
        # -> tVec va fi in metri
        rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(
            corners, self.marker_size_m, self.camera_matrix, self.dist
        )

        poses = []
        for i, corner in enumerate(corners):
            # tVecs[i] este vectorul [x, y, z] in metri
            tvec_m = tVecs[i][0]
            rvec = rVecs[i][0]


            # Distanta
            distance_m = np.linalg.norm(tVecs[i])
            distance_cm = distance_m * 100.0 # Conversie pt afisare


            # Unghi (rotatie in plan 2D)
            marker_corners = corner.reshape((4, 2))
            # Top-Right Top-Left Bottom-Right Bottom-Left
            tl, tr, br, bl = marker_corners
            delta = tr - tl
            angle = np.degrees(np.arctan2(delta[1], delta[0]))
            
            poses.append({
                "ids": int(ids[i][0]),
                "corners": marker_corners,
                "rVec": rVecs[i],
                "tVec": tVecs[i],
                "distance_cm": distance_cm,
                "angle": angle,
                "x_cm": tvec_m[0] * 100,
                "y_cm": tvec_m[1] * 100
            })
        return poses

    
    
    def ret_dimensions(self, frame):
        """
        Returneaza mm per pixel, unghiul markerului si colturile,
        pentru markerul #1 gasit.
        """
        corners, ids, _ = self.detect(frame)
        if ids is None or len(corners) == 0:
            return None
        
        marker_corners = np.array(corners[0]).reshape((4, 2))
        (tl, tr, br, bl) = marker_corners

        # Calculam latime si inaltime in pixeli
        px_w = np.linalg.norm(tr - tl)
        px_h = np.linalg.norm(tl - bl)

        # Medie pentru stabilizare
        avg_mark_px = (px_w + px_h) / 2.0

        # Dimensiunea reala a markerului in mm
        real_mark_mm = self.marker_size_m * 1000.0
        mm_per_px = real_mark_mm/ avg_mark_px

        delta = tr - tl
        angle = np.degrees(np.arctan2(delta[1], delta[0]))

        return{
            "mm_per_px": mm_per_px,
            "angle": angle,
            "corners": marker_corners
        }
    
    


    # ====================================================================
    #   Utilitati desenare
    # ====================================================================
    def draw_marker_info(self, frame, poses: List[Dict[str, Any]]):
        """Deseneaza contur, ID, distanta, axe XYZ pe frame."""
        # if pose is None:
        #     return
        
        for pose in poses:
            rVec = pose["rVec"]
            tVec = pose["tVec"]
            if rVec is None or tVec is None:
                #logger.info(f"Skipping marker {pose['ids']} due to missing pose data.")
                continue

            corners = pose["corners"].astype(np.int32)
            marker_id = pose["ids"]

            # Desenare contur marker
            cv.polylines(frame, [corners], True, (0, 255, 255), 4, cv.LINE_AA)

            top_right, top_left, bottom_right, bottom_left = corners

            # Desenare axe
            distance = pose["distance_cm"]
            axis_len =  self.marker_size_m * 0.5
            try:
                cv.drawFrameAxes(frame, 
                                self.camera_matrix, 
                                self.dist, 
                                rVec, 
                                tVec, 
                                axis_len)
            except Exception:
                pass


            # Desenare ID distanta
            cv.putText(
                frame,
                f"id: {marker_id} Dist: {distance}cm",
                tuple(top_right),
                cv.FONT_HERSHEY_PLAIN,
                1.3, (200,100,0), 2, cv.LINE_AA)


            # COORDONATE X,Y
            cv.putText(
                frame,
                f"x:{round(tVec[0][0],1)} y:{round(tVec[0][1],1)}",
                tuple(bottom_right),
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (200,100,0),
                2,
                cv.LINE_AA
            )

  



def _ensure_marker_size_m(marker_size: float, unit: str = "cm") -> float:
    unit = unit.lower()
    if unit.startswith("c"):
        return float(marker_size) / 100.0
    if unit.startswith("m"):
        return float(marker_size)
    raise ValueError("Unitate de masura gresita 'cm' sau 'm'.")












####        ALTE FUNCTII NEUTILIZATE

def project_to_marker_plane(pt_img: Tuple[float, float], camera_matrix, rvec, tvec):
    """
    Proiectează o singură coordonată de pixel a imaginii (x,y) pe planul markerului 
    (Z=0 în cadrul markerului).
    Returnează punctul 3D în **milimetri** (X_mm, Y_mm, Z_mm ~ 0).
    
    Rezumat matematic:
    - Fie R, t poziția markerului în coordonatele camerei: X_cam = R * X_marker + t
    - Vectorul pixelilor (u = K^-1 * [x, y, 1]^T) este direcția în coordonatele camerei
    - Găsim lambda astfel încât (R * X_marker + t)[2] == 0 (planul markerului Z_marker = 0)
    - Rezolvă pentru X_marker

    Notă:
    rvec: Vector de rotație Rodrigues din estimatePose (3x1)
    tvec: Vector de translație din estimatePose (3x1) în metri
    """
    # Matrice rotatie
    R, _= cv.Rodrigues(rvec)

    #Translatie
    t = tvec.reshape(3, 1)

    # Rotatie + translatie
    Rt = np.hstack([R, t])

    # Inversăm transformarea 3D → 2D pentru un plan Z=0
    # Ecuația: λ * p_img = K * (R*[X,Y,0]^T + t)
    # => [X,Y,0]^T = R_inv * (K_inv * λ*p_img - t)

    K_inv = np.linalg.inv(camera_matrix)
    R_inv = np.linalg.inv(R)

    p = np.array([pt_img[0], pt_img[1], 1.0], dtype=float).reshape(3, 1)

    ray = K_inv @ p

    lmda = -t[2, 0] / (R[2] @ ray)[0]

    p_cam = lmda * ray
    p_real = R_inv @ (p_cam - t)

    return p_real.flatten()


def project_points_to_plane(self, pts_img: List[Tuple[float, float]], rvec, tvec):
    """
    Maps a list of image pixel points to 3D coordinates on marker plane (mm).
    Returns Nx3 array.
    """
    pts3 = []
    for p in pts_img:
        pts3.append(project_to_marker_plane(p, rvec=rvec, tvec=tvec))
    return np.vstack(pts3)


def real_distance_between_points(p1_img, p2_img, camera_matrix, rvec, tvec):
    P1 = project_to_marker_plane(p1_img, camera_matrix, rvec, tvec)
    P2 = project_to_marker_plane(p2_img, camera_matrix, rvec, tvec)
    return np.linalg.norm(P1 - P2)  # distanță reală în mm



def compute_mm_per_px(marker_corners_2d, camera_matrix, rvec, tvec):
    pts_real = []
    for pt in marker_corners_2d:
        pts_real.append(project_to_marker_plane(pt, camera_matrix, rvec, tvec))

    # 4 colțuri reale 3D -> măsurăm distanța reală între ele
    d_real = np.linalg.norm(pts_real[0] - pts_real[1])  
    
    d_px = np.linalg.norm(marker_corners_2d[0] - marker_corners_2d[1])

    return d_real / d_px  # mm/pixel (real)

def compute_mm_per_px(self, marker_corners, rvec, tvec):
    if marker_corners is None or len(marker_corners) == 0:
            raise ValueError("marker_corners_img required")

    # project first two adjacent corners (tl, tr) to plane
    # ensure ordering consistent with estimatePose (OpenCV usually returns [tl,tr,br,bl])
    tl = tuple(map(float, marker_corners[0]))
    tr = tuple(map(float, marker_corners[1]))

    P_tl = self.project_pixel_to_marker_plane(tl, rvec, tvec)  # mm
    P_tr = self.project_pixel_to_marker_plane(tr, rvec, tvec)  # mm

    real_dist_mm = float(np.linalg.norm(P_tr[:2] - P_tl[:2]))  # mm
    px_dist = float(np.linalg.norm(np.asarray(tr) - np.asarray(tl)))

    if px_dist == 0:
        raise ValueError("Zero pixel distance between corners")

    mm_per_px = real_dist_mm / px_dist
    return mm_per_px



def project_pixel(self, pt_img, pose):
    return project_to_marker_plane(pt_img, self.camera_matrix, pose["rvec"], pose["tvec"])




def draw_marker_info(self, frame: np.ndarray, poses: List[Dict[str, Any]], draw_axes_len_m: float = None):
        """
        Draw contours, ids, axes and info for each pose.
        draw_axes_len_m: length of axes in meters (if None, uses marker_size/2)
        """
        if poses is None:
            return

        # default axes len
        for p in poses:
            rvec = p["rvec"]
            tvec = p["tvec"]
            corners = p["corners"].astype(int)
            marker_id = p["id"]

            # draw polygon
            cv.polylines(frame, [corners.reshape((-1, 2))], True, (0, 255, 255), 2, cv.LINE_AA)
            # label
            cv.putText(frame, f"ID:{marker_id}", tuple(corners[0]), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)

            # draw frame axes
            length_m = draw_axes_len_m if draw_axes_len_m is not None else (self.marker_size_m / 2.0)
            try:
                cv.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, length_m)
            except Exception:
                # ensure we don't crash the draw
                pass

def draw_pose_text(self, frame: np.ndarray, poses: List[Dict[str, Any]]):
    """Put text info (distance mm, angle) near marker"""
    if poses is None:
        return
    for p in poses:
        corners = p["corners"].astype(int)
        dist_mm = p.get("distance_mm", None)
        angle = p.get("angle_deg", 0.0)
        txt = f"{dist_mm:.1f} mm, {angle:.1f} deg" if dist_mm is not None else f"{angle:.1f} deg"
        cv.putText(frame, txt, tuple(corners[1]), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
