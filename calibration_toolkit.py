# calibrate_pipeline.py
import argparse
import time
import os
import cv2 as cv

from Camera import CameraStream
from Calibration import ChessboardImageSaver, CircleImageSaver
from Calibration import CameraCalibrator


try:
    #from Camera import FLIRCamera  # dacă ai acest modul
    from Camera import WebCam
except Exception:
    FLIRCamera = None
    WebCam = None
    print("Warning: FLIRCamera not found. You must import your camera class manually.")

def run_chessboard_flow(camera_obj, board_dim=(9,6), square_size=25.0, capture_count=15):
    stream = CameraStream(camera_obj)
    stream.start()
    saver = ChessboardImageSaver(stream, board_dim=board_dim)
    print("Press 's' to save each good frame, 'q' to stop early.")
    saver.start_capture()
    # after capture stop, run calibrator
    stream.stop()

def run_calibration(images_folder="calibration-images", pattern="chessboard", board_dim=(9,6), square_size=25.0):
    cal = CameraCalibrator(pattern_type=pattern, board_dim=board_dim, square_size=square_size,
                           images_folder=images_folder, filename="MultiMatrix.npz", show_results=True)
    cal.calibrate()
    cal.calc_reprojection_error()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["capture", "calibrate", "all"], default="all")
    parser.add_argument("--pattern", default="chessboard")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square", type=float, default=25.0)
    args = parser.parse_args()

    board_dim = (args.cols, args.rows)

    # create camera object (adapt this)
    cam = None
    if WebCam is not None:
        cam =WebCam(width=1280, height=720)   # adaptează parametrii aici dacă trebuie
    else:
        raise RuntimeError("No camera available. Implement FLIRCamera or change imports.")

    if args.mode in ("capture", "all"):
        # Start capture UI
        stream = CameraStream(cam)
        stream.start()
        saver = ChessboardImageSaver(stream, board_dim=board_dim)
        saver.start_capture()
        stream.stop()

    if args.mode in ("calibrate", "all"):
        run_calibration(images_folder="calibration-img", pattern=args.pattern, board_dim=board_dim, square_size=args.square)



