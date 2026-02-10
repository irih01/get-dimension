from Process import DetectAruco, ObjDetector
from Camera import CameraManager
import cv2 as cv



def main():
    manager = CameraManager()
    #cam = manager.create_camera("webcam", cam_id=0, width=1280, height=720)
    flir = manager.create_camera("flir",cam_id=0, image_format="BGR8")
    stream = manager.create_stream(flir)
    aruco = DetectAruco(marker_size=5.2, strict=False)
    obj = ObjDetector()
    manager.start()
    print("stream pornit")

    print("aruco creat")
    try:
        
        while True:
            
            _, frame = manager.get_frame()
            
            
            
            if frame is not None:
                
                poses = aruco.estimate_pose(frame)
                
                if poses:
                    aruco.draw_marker_info(frame, poses)
                    result = obj.process(frame)

                    if result["ok"]:
                        print(result["width_mm"], result["height_mm"])

                cv.imshow("Detections", frame)
               
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        manager.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
