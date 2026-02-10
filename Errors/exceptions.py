
class SIRBaseError(Exception):
    """Eroare generala pentru proiect"""

class CameraError(SIRBaseError):
    pass

class CalibrationError(SIRBaseError):
    pass

class MarkerError(SIRBaseError):
    pass

class ProcessingError(SIRBaseError):
    pass

class CalibrationMissingError(RuntimeError):
    pass
