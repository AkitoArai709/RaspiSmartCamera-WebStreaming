"""camera.py
    Summay: 
        Camera class
        Take a camera capture of the RasPi and detect sleepiness.
"""

import cv2
from baseCamera import BaseCamera
from detectionSleepiness import DetectionSleepiness

class Camera(BaseCamera):
    """ Camera class.

    Args:
        BaseCamera (BaseCamera): Base camera class.
    """
    tick = 0
    fpsColor = (0, 255, 0)

    def __init__(self):
        super().__init__()

    @staticmethod
    def frames():
        """ Camera frames.

        Raises:
            RuntimeError: Can not open camera.

        Yields:
            byte: Camera frame.
        """
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, frame = camera.read()

            frame = DetectionSleepiness.getDetectResultFrame(frame)
            yield cv2.imencode('.jpg', Camera.__drawingFps(frame))[1].tobytes()
            
    @staticmethod
    def __drawingFps(frame):
        """ Calutulation fps and drawing to image frame. 

        Args:
            img (ndaaray): Camera frame.

        Returns:
            ndaaray: Camera frame drawing FPS.
        """
        fps = 0
        if Camera.tick != 0:
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - Camera.tick)
        Camera.tick = cv2.getTickCount()
        return cv2.putText(frame, "FPS:{} ".format(int(fps)), 
                    (520, 30), cv2.FONT_HERSHEY_DUPLEX, 1, Camera.fpsColor, 1, cv2.LINE_AA)
