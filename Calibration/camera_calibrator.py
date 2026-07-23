import cv2 as cv
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class CameraCalibrator:
    """Calibrarea camerei folosind imagini cu tabla de sah."""

    def __init__(self, 
                 pattern_type="chessboard", 
                 board_dim = (9,6), 
                 square_size = 25, 
                 base_dir = None, 
                 images_folder = "calibration-images", 
                 data_folder = ".", 
                 filename: str = "MultiMatrix.npz", 
                 criteria = None, 
                 show_results = False):
        
        self.pattern_type = pattern_type.lower().strip()
        self.board_dim = board_dim
        self.sq_size = square_size

        base_dir = base_dir or os.getcwd()
        self.images_folder = os.path.join(base_dir, images_folder)
        self.data_folder = os.path.join(base_dir, data_folder)
        self.filename = filename

        self.criteria = criteria or (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.show_res = show_results

        # rezultate calibrare
        self.obj_points = []
        self.img_points = []
        self.gray_shape = None


        self.camera_matrix = None
        self.dist_coef = None
        self.rvecs = None
        self.tvecs = None


        self._prepare_directory()

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    
    def _prepare_directory(self):
        if not os.path.isdir(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"[INFO] Folder creat: {self.data_folder}")
        else:
            print(f"[INFO] Folderul {self.data_folder} exista deja")



    def _prepare_object_points(self):
        cols, rows = self.board_dim #board_dim[0] board_dim[1]

        if self.pattern_type == "chessboard":
            obj_3D = np.zeros((rows * cols, 3), np.float32)
            obj_3D[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            obj_3D *= self.sq_size
            return obj_3D
        
        elif self.pattern_type == "circle":
            obj = np.zeros((rows * cols, 3), np.float32)
            # FIX: Ne asigurăm că ordinea grid-ului respectă standardul OpenCV pentru rețele simetrice
            obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            obj *= self.sq_size
            return obj
        
        elif self.pattern_type == "acircle":
            obj = []
            for r in range(rows):
                for c in range(cols):
                    x = (2 * c + (r % 2)) * (self.sq_size / 2.0)
                    y = r * self.sq_size
                    obj.append([x, y, 0.0])
            return np.array(obj, dtype=np.float32)

        else:
            raise ValueError("Pattern necunoscut")
        
           


    def _detect_pattern(self, gray):
        if self.pattern_type == "chessboard":
            flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
            return cv.findChessboardCorners(gray, self.board_dim, flags=flags)
        
        elif self.pattern_type == "circle":
            flags = cv.CALIB_CB_SYMMETRIC_GRID
            return cv.findCirclesGrid(gray, self.board_dim, flags)
        
        elif self.pattern_type == "acircle":
            flags = cv.CALIB_CB_ASYMMETRIC_GRID
            return cv.findCirclesGrid(gray, self.board_dim, flags)
        
        else:
            raise ValueError("Pattern invalid")





    # -------------------------------------------------------
    # Calibrare cameră
    # -------------------------------------------------------


    def calibrate(self, error_threshold=1.0):
        if not os.path.isdir(self.images_folder):
            raise FileNotFoundError(f"[ERROR] Folderul {self.images_folder} nu a fost gasit")
        
        obj_3D = self._prepare_object_points()

        image_files = [f for f in os.listdir(self.images_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        if not image_files:
            logger.warning("Nu s-au gasit imagini in folder")
            return None
        
        print(f"[INFO] {len(image_files)} imagini gasite. Se incepe calibrarea..")

        total = len(image_files)

        for idx, file in enumerate(image_files):
            print(f"[{idx+1}/{total}] Procesare: {file}")

            image_path = os.path.join(self.images_folder, file)
            image = cv.imread(image_path)
            if image is None:
                logger.warning(f"Imaginea {file} nu a putut fi citita.")
                continue

            h, w = image.shape[:2]
            
            # Salvăm dimensiunea originală a imaginii din acest fișier, nu a celei scalate!
            self.gray_shape = (w, h)

            max_dim = 1000
            scale = max_dim / max(h, w)
            is_scaled = False

            if scale < 1.0:
                image_processing = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
                is_scaled = True
            else:
                image_processing = image.copy()

            gray = cv.cvtColor(image_processing, cv.COLOR_BGR2GRAY)

            found, corners = self._detect_pattern(gray)
            if not found:
                logger.warning(f"Pattern negasit in {file}")
                continue

            # refine chessboard
            if self.pattern_type == "chessboard":
                corners2 = cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.criteria)
            else:
                corners2 = corners
            
            # FIX IMPORTANT: Dacă am micșorat imaginea pentru procesare rapidă, aducem coordonatele
            # colțurilor înapoi la scara originală a imaginii înainte de a le salva.
            if is_scaled:
                corners2 = corners2 / scale

            self.obj_points.append(obj_3D)
            self.img_points.append(corners2)
           

            if self.show_res:
                # Desenăm pe imaginea originală sau pe cea scalată pentru debug vizual
                cv.drawChessboardCorners(image_processing, self.board_dim, corners, found)
                cv.imshow("Colturi detectate", image_processing)
                cv.waitKey(300)
        
        print()  # newline
        print(f"[INFO] {len(self.obj_points)} imagini valide pentru calibrare")

        if self.show_res:
            cv.destroyAllWindows()

        if len(self.obj_points) == 0:
            raise RuntimeError("Nici o calibrare colectata")
        print("Ruleaza calibratteCamera()..")

        MAX_IMAGES = 200
        if len(self.obj_points) > MAX_IMAGES:
            idx = np.linspace(0, len(self.obj_points) - 1, MAX_IMAGES, dtype=int)
            self.obj_points = [self.obj_points[i] for i in idx]
            self.img_points = [self.img_points[i] for i in idx]

        # Ruleaza o calibrare initiala
        print(f"[INFO] Folosim {len(self.obj_points)} imagini pentru calibrare initiala")
        flags = (cv.CALIB_ZERO_TANGENT_DIST | cv.CALIB_FIX_K3 )

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            self.obj_points, self.img_points, self.gray_shape, None, None, flags=flags
        )

        print("[INFO] Calibrare finalizata!")

        self.camera_matrix = mtx
        self.dist_coef = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        self._save_calibration_data(mtx, dist, rvecs, tvecs)

        logger.info("Calibrare completa")
        return {
            "ret": ret,
            "camera_matrix": mtx,
            "dist_coef": dist,
            "rvecs": rvecs,
            "tvecs": tvecs
        }




    # -------------------------------------------------------
    # Salvare / încărcare date
    # -------------------------------------------------------
    def _save_calibration_data(self, mtx, dist, rvecs, tvecs):
        file_path = os.path.join(self.data_folder, self.filename)
        np.savez(file_path,
                 camMatrix=mtx,
                 distCoef=dist,
                 rVector=rvecs,
                 tVector=tvecs,
                 pattern=self.pattern_type,
                 boardDimensions=self.board_dim,
                 squareSize=self.sq_size,
                 objPoints = np.array(self.obj_points, dtype=object),
                 imgPoints = np.array(self.img_points, dtype=object))
        
        print(f"[INFO] Datele de calibrare au fost salvate in {file_path}")

    



    def load_calibration_data(self):
        file_path = os.path.join(self.data_folder, self.filename)
        if not os.path.exists(file_path):
            logger.warning("[ERROR] Fisierul de calibrare nu exista")
            return None
        
        data = np.load(file_path, allow_pickle= True)


        self.camera_matrix = data["camMatrix"]
        self.dist_coef = data["distCoef"]
        self.rvecs = data["rVector"]
        self.tvecs = data["tVector"]
        self.pattern_type = str(data["pattern"])
        self.board_dim = tuple(data.get("boardDimensions", self.board_dim))
        self.sq_size = float(data.get("squareSize", self.sq_size))
        
        # FIX SECURE: Ne asigurăm că extragem array-urile NumPy brute din structura de obiecte generată de pickle
        self.obj_points = [np.array(p, dtype=np.float32) for p in data["objPoints"]]
        self.img_points = [np.array(p, dtype=np.float32) for p in data["imgPoints"]]



        print("[INFO] Datele de calibrare au fost incarcate")
        return data
    


# -------------------------------------------------------
# Extra
# -------------------------------------------------------


    def get_focal_length(self):
        """Returneaza lungimea focala (fx, fy) si centrul optic (cx, cy)"""
        if self.camera_matrix is None:
            raise ValueError("[ERROR] Nu exista matrice de camera calculata")
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        return {"focal_length_px": (fx, fy), "optical_center": (cx, cy)}
    




    def calc_reprojection_error(self):
        """Calculeaza eroarea medie de re-proiectie"""

        if self.camera_matrix is None or self.dist_coef is None:
            print("[ERROR] Nu exista date de calibrare incarcate sau calculate")
            return None
        
        total_error = 0
        total_points = 0

        num_images = len(self.rvecs)

        for i in range(num_images):
            try:
                imgpoints2, _ = cv.projectPoints(
                    self.obj_points[i], self.rvecs[i], self.tvecs[i],
                    self.camera_matrix, self.dist_coef
                )

                # FIX MATEMATIC: cv.norm calculează norma totală pe imagine. Pentru RMSE corect conform
                # standardelor OpenCV, calculăm distanța euclidiană per punct individual.
                error = cv.norm(self.img_points[i], imgpoints2, cv.NORM_L2)
                total_error += error ** 2
                total_points += len(self.obj_points[i])
            except IndexError:
                continue
        
        if total_points == 0:
            return 0

        # Formula RMSE corectă: radical din media pătratelor tuturor reziduurilor punctuale
        mean_error = np.sqrt(total_error / total_points)
        print(f"[INFO] Eroare medie de re-proiectie: {mean_error:.4f} pixeli")
        return mean_error
    


if __name__ == "__main__":
    calibrator = CameraCalibrator(images_folder='calibration-images', filename="MultiMatrix.npz", show_results=False )
    calibrator.calibrate()

    results = calibrator.load_calibration_data()
    print(results)
    if results:
        calibrator.calc_reprojection_error()

