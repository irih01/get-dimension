```bash
BaseCamera (abstract)
   ↳ FLIRCam
   ↳ WebCam
CameraManager
CameraStream 


+--------------------+
|    CameraManager   |  -> decide camera activă
+---------+----------+
          |
          v
+--------------------+
|     BaseCamera     |  -> API unificat
+--------------------+
     /            \
    v              v
FLIRCam       WebCam
    \            /
     +----------+
          |
          v
+--------------------+
|    CameraStream    |  -> produce flux constant pentru GUI
+--------------------+
          |
          v
      GUI / Processing


```
     

